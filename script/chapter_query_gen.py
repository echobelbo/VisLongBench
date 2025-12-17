import json
import os
import re
from tqdm import tqdm
from openai import OpenAI

dataset="tutorial"

structure_path = f"./data/{dataset}/chunk.json"
summary_folder = f"./data/{dataset}/summaries_json"
output_path = f"./data/{dataset}/query/chapter_queries_ori.json"
api_model = "gpt-4o"
api_key = "sk-CJ4dP8IEf6IM9INBy9CVHFoh65xkC5Zd7A0LV5xrGiGGY6Sj"
client = OpenAI(api_key=api_key, base_url="https://chatapi.onechats.top/v1")


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


def load_structure(path: str):
    with open(path, "r") as f:
        return json.load(f)


def load_summary(summary_folder: str, ppt_name: str):
    summary_file = os.path.join(summary_folder, f"{ppt_name}.json")
    if not os.path.exists(summary_file):
        print(f"⚠️ Summary file not found for {ppt_name}")
        return None
    with open(summary_file, "r") as f:
        return json.load(f)


def generate_qa_for_segment(client, api_model, combined_summary: str, query_num: int):
    prompt = f"""
You are a professional business analysis assistant. Below is a summary of a chapter from a business report, based on the content of multiple slides.

\"\"\" 
{combined_summary}
\"\"\"


Your task is to generate **{query_num} high-level Q&A pairs** that help a reader understand and reflect on the main ideas of this chapter.

### Requirements:

#### For each question:
- Focus on a major theme, trend, or insight from the summary;
- Avoid specific slide-level details;
- Encourage analytical thinking and structured understanding;
- Be clearly written and professional.

#### For each answer:
- Be concise and accurate;
- Synthesize relevant information from the summary;
- Avoid directly copying long phrases.

### Output Format (in JSON):
Return a JSON array of objects, each with a "question" and an "answer" field. Do **not** include any explanatory text.

Example:
[
  {{
    "question": "What are the main market forces driving the growth of sector X?",
    "answer": "The growth is primarily driven by increased consumer demand, regulatory support, and technological advancements."
  }},
  ...
]

Now generate the Q&A pairs in **valid JSON format**:
"""
    try:
        response = client.chat.completions.create(
            model=api_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        generated_questions = response.choices[0].message.content
        return extract_json(generated_questions)
    except Exception as e:
        print(f"❌ GPT调用失败: {e}")
        return "ERROR"


def process_ppt(client, api_model, ppt_name: str, segments: list, summary_folder: str):
    summary_data = load_summary(summary_folder, ppt_name)
    if summary_data is None:
        return None

    page_summary_map = {s["page"]: s["summary"] for s in summary_data.get("slides", [])}
    ppt_result = []

    for seg in tqdm(segments, desc=f"Processing segments for {ppt_name}"):
        start = seg.get("start")
        end = seg.get("end")
        title = seg.get("title", "")

        if "foreword" in title.lower():
            print(f"⚠️ Skipping foreword segment for {ppt_name} - {title}")
            continue

        length = end - start + 1 if end is not None and start is not None else 0
        if length <= 0:
            print(f"⚠️ Invalid segment length for {ppt_name} segment {title}")
            continue

        if length <= 20:
            query_num = 3
        elif length <= 30:
            query_num = 5
        else:
            query_num = 10

        segment_summaries = [page_summary_map.get(p, "") for p in range(start, end + 1)]
        combined_summary = "\n".join([s for s in segment_summaries if s.strip()])

        if not combined_summary.strip():
            print(f"⚠️ No summaries found for {ppt_name} segment {title}")
            continue

        questions = generate_qa_for_segment(client, api_model, combined_summary, query_num)

        ppt_result.append({
            "title": title,
            "start": start,
            "end": end,
            "questions": questions
        })

    return ppt_result


def chunk_query_gen_main():
    """
    主流程：
    1. 从 structure_path 加载目标结构
    2. 若 output_path 存在，加载已有结果跳过已完成项
    3. 每完成一个 PPT 自动保存
    4. 可断点续跑
    """

    # 加载结构文件
    with open(structure_path, "r", encoding="utf-8") as f:
        structure = json.load(f)

    # 如果已存在部分结果，则加载
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            final_queries = json.load(f)
        print(f"🔄 检测到已有结果文件，已加载 {len(final_queries)} 个已完成的PPT。")
    else:
        final_queries = {}

    # 统计需要跳过的
    completed = set(final_queries.keys())

    for ppt_name, segments in tqdm(structure.items(), desc="Processing PPTs"):
        if ppt_name in completed:
            print(f"⏭️ 跳过已完成: {ppt_name}")
            continue

        try:
            ppt_result = process_ppt(client, api_model, ppt_name, segments, summary_folder)
            if ppt_result is not None:
                final_queries[ppt_name] = ppt_result
                print(f"✅ 完成 {ppt_name} 的 query 生成")

                # 每完成一个PPT即保存
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(final_queries, f, indent=2, ensure_ascii=False)
                print(f"💾 已保存进度 ({len(final_queries)}/{len(structure)})")

        except Exception as e:
            print(f"❌ 处理 {ppt_name} 时出错: {e}")

    print(f"✅ 所有 query 已保存到 {output_path}")
    return final_queries


def load_chunk_query_gen(query_ori_path: str):
    """加载并返回 chunk_query_gen_main 函数"""
    with open(query_ori_path, "r") as f:
        queries = json.load(f)
    return queries
if __name__ == "__main__":
    chunk_query_gen_main()
    # queries_ori = load_chunk_query_gen(output_path)   