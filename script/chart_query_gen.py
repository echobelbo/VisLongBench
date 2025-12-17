import json
import os
import base64
import sys
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt.query_prompt import direct_prompt, detail_prompt


# ---------------------------
# JSON 清理工具
# ---------------------------
def extract_json(select_response: str):
    cleaned = select_response.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    matches = re.findall(r'(\{.*?\}|\[.*?\])', cleaned, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    raise ValueError("No valid JSON content could be extracted.")


# ---------------------------
# 子进程执行函数（处理一个 PPT）
# ---------------------------
def process_single_ppt(ppt_name, segments, ppt_image_root, prompt, api_key, base_url, model):
    """
    子进程内部执行：处理单个 PPT 的所有 segment
    """

    client = OpenAI(api_key=api_key, base_url=base_url)

    def encode_image(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_messages(category, images_b64):
        return [
            {"role": "system", "content": "You are an AI that generates analytical questions for a given PPT image set."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt["diff"][category].format(query_num=prompt["num"])}
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} 
                    for b64 in images_b64
                ]
            }
        ]

    ppt_dir = os.path.join(ppt_image_root, ppt_name)
    if not os.path.exists(ppt_dir):
        print(f"❌ 找不到目录: {ppt_dir}")
        return ppt_name, []

    ppt_result = []

    for seg in segments:
        group_id = seg["group_id"]
        start_page = seg["start_page"]
        end_page = seg["end_page"]
        category = seg["category"]

        if category not in prompt["diff"]:
            print(f"⚠️ 未知类别 {category} 跳过。")
            continue

        # 读取该段的所有图片
        images = []
        for page in range(start_page, end_page + 1):
            img_path = os.path.join(ppt_dir, f"page_{page}.jpg")
            if os.path.exists(img_path):
                images.append(encode_image(img_path))
            else:
                print(f"⚠️ 找不到图片: {img_path}")

        if not images:
            continue

        # 构建 messages
        messages = build_messages(category, images)

        # 模型调用
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            data = extract_json(response.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ API 调用失败: {e}")
            continue

        # 格式统一成 list
        if isinstance(data, dict) and "question" in data:
            data = [data]

        # 写入结果
        for i, q in enumerate(data):
            ppt_result.append({
                "question_id": f"{group_id}_{i+1}",
                "start": start_page,
                "end": end_page,
                "category": category,
                "question": q["question"],
                "answer": q["answer"],
                "difficuty": prompt["difficuty_text"]
            })

    return ppt_name, ppt_result


# ---------------------------
# 主流程：并行执行
# ---------------------------
class PPTQueryGeneratorParallel:

    def __init__(self, api_key, base_url="https://chatapi.onechats.top/v1", model="gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def process_ppt_segments(self, ppt_targets, ppt_image_root, prompt, output_path, max_workers=4):

        # 加载已有结果（断点续跑）
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"🔁 加载已完成：{len(results)} 个 PPT")
        else:
            results = {}

        remaining = [p for p in ppt_targets if p not in results]
        print(f"📌 总 {len(ppt_targets)} 个 PPT，待处理 {len(remaining)} 个")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_ppt,
                    ppt_name,
                    ppt_targets[ppt_name],
                    ppt_image_root,
                    prompt,
                    self.api_key,
                    self.base_url,
                    self.model
                ): ppt_name
                for ppt_name in remaining
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PPTs"):
                ppt_name = futures[future]
                try:
                    name, ppt_result = future.result()
                    results[name] = ppt_result
                except Exception as e:
                    print(f"❌ 处理失败 {ppt_name}: {e}")
                    continue

                # 每个 PPT 处理完立即保存一次
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"🎉 全部处理完成 → {output_path}")
        return results


# ---------------------------
# main
# ---------------------------
def main():

    api_key = "sk-65xkC5Zd7A0LV5xrGiGGY6Sj"

    dataset = "tutorial"
    json_path = f"./data/{dataset}/query/chart_label.json"
    ppt_image_root = f"./data/{dataset}/images"

    difficuty_text = "direct"
    output_path = f"./data/{dataset}/query/{difficuty_text}_queries_ori.json"

    if difficuty_text == "direct":
        diff = direct_prompt
        query_num = 1
    elif difficuty_text == "detail":
        diff = detail_prompt
        query_num = 3

    prompt = {
        "diff": diff,
        "num": query_num,
        "difficuty_text": difficuty_text
    }

    # load target json
    with open(json_path, "r", encoding="utf-8") as f:
        ppt_targets = json.load(f)

    generator = PPTQueryGeneratorParallel(api_key=api_key)

    generator.process_ppt_segments(
        ppt_targets,
        ppt_image_root,
        prompt,
        output_path,
        max_workers=4
    )


if __name__ == "__main__":
    main()
