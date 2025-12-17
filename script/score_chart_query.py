import json
import os
import base64
import sys
import re
import time
import random
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt.query_prompt import score_chart_prompt


# -------------------------- JSON CLEAN --------------------------
def extract_json(select_response: str):
    """清理并解析模型返回的JSON字符串"""
    cleaned = select_response.strip().replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    json_matches = re.findall(r'(\{.*?\}|\[.*\])', cleaned, re.DOTALL)
    for match in json_matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    raise ValueError("No valid JSON content could be extracted.")


# ----------------------------- MAIN CLASS -----------------------------
class ScoreGenerator:
    def __init__(self, api_key, base_url="https://chatapi.onechats.top/v1", model="gemini-2.5-pro"):
        # ⚠️ 只能初始化一次，否则多进程/线程不能 pickle
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def encode_image_to_base64(self, img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def build_messages(self, question, answer, images_b64, prompt):
        messages = [
            {
                "role": "system",
                "content": "You are an AI that evaluates Q/A quality based on charts."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.format(question=question, answer=answer)}
                ]
            }
        ]

        for img_b64 in images_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        return messages

    # --------------------- 统一 API 调用 + 限流重试 -----------------------
    def safe_generate(self, messages, max_tokens=1500, retry=6):
        for i in range(retry):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return resp.choices[0].message.content

            except Exception as e:
                err = str(e)
                print(f"\n⚠️ API 调用失败：{err}")

                # —— 限流 429 特殊处理 ——
                if "429" in err or "rate" in err.lower():
                    sleep_time = 1.5 * (2 ** i) + random.random()
                    print(f"⏳ 触发限流，等待 {sleep_time:.2f}s 后重试...")
                    time.sleep(sleep_time)
                    continue

                # —— 额度不足，退出 ——
                if "quota" in err.lower() or "insufficient" in err.lower():
                    print("❌ 检测到额度不足，程序终止。")
                    sys.exit(1)

                # 其他异常短暂等待
                time.sleep(1)

        return "[ERROR] Retry Failed"

    # ------------------------- 单任务处理 -------------------------
    def _process_single(self, ppt_name, seg, ppt_dir, prompt):
        question_id = seg["question_id"]
        start_page = seg["start"]
        end_page = seg["end"]
        question = seg["question"]
        answer = seg["answer"]
        category = seg["category"]

        # 读取所有图片
        images = []
        for p in range(start_page, end_page + 1):
            path = os.path.join(ppt_dir, f"page_{p}.jpg")
            if os.path.exists(path):
                images.append(self.encode_image_to_base64(path))

        if not images:
            return None

        # 组装消息
        messages = self.build_messages(question, answer, images, prompt)

        # 调用模型
        response = self.safe_generate(messages)

        if response.startswith("[ERROR]"):
            return None

        try:
            json_obj = extract_json(response)
        except:
            return None

        # ------- 打分合理性修正 -------
        if json_obj["answerable_without_slides"] == 0 or \
           json_obj["clarity"] + json_obj["relevance"] + json_obj["usefulness"] < 5:
            json_obj["answerable_without_slides"] = 0

        return {
            "question_id": question_id,
            "start": start_page,
            "end": end_page,
            "category": category,
            "question": question,
            "answer": answer,
            "score": {
                "answerable_without_slides": json_obj["answerable_without_slides"],
                "clarity": json_obj["clarity"],
                "relevance": json_obj["relevance"],
                "usefulness": json_obj["usefulness"]
            }
        }

    # --------------------- 并发调度 ---------------------
    def process_ppt_segments(self, ori_qa, ppt_image_root, prompt, output_path):

        results = {}
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)

        max_workers = 10  # ← 你可以改成 8~12

        for ppt_name, segments in tqdm(ori_qa.items(), desc="Processing PPT", unit="ppt"):
            ppt_dir = os.path.join(ppt_image_root, ppt_name)

            if ppt_name in results:
                continue

            print(f"\n▶ 处理 PPT：{ppt_name}")
            results[ppt_name] = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for seg in segments:
                    futures.append(
                        executor.submit(
                            self._process_single, ppt_name, seg, ppt_dir, prompt
                        )
                    )

                for future in tqdm(as_completed(futures), total=len(futures)):
                    item = future.result()
                    if item:
                        results[ppt_name].append(item)

            # 保存中间结果
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results


# ============================== MAIN ==============================
if __name__ == "__main__":
    api_key = "sk-CJ4dP8IEf6IM9INBy9CVHFoh65xkC5Zd7A0LV5xrGiGGY6Sj"
    dataset = "tutorial"
    ppt_image_root = f"./data/{dataset}/images"
    difficuty_text = "direct"

    ori_path = f"./data/{dataset}/query/{difficuty_text}_queries_ori.json"
    final_path = f"./data/{dataset}/query/{difficuty_text}_queries.json"

    with open(ori_path, "r", encoding="utf-8") as f:
        ori_qa = json.load(f)

    generator = ScoreGenerator(api_key=api_key)

    results = generator.process_ppt_segments(
        ori_qa, ppt_image_root, score_chart_prompt, final_path
    )

    print(f"✅ 生成完成，结果已保存到 {final_path}")
