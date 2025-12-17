import json
import re
from tqdm import tqdm
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt.query_prompt import score_chapter_prompt


# -----------------------------------------
# JSON 解析工具
# -----------------------------------------
def extract_json(select_response: str):
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


def format_qa_pairs(qa_list):
    formatted = ""
    if isinstance(qa_list, str):
        return ""
    for i, qa in enumerate(qa_list, 0):
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        formatted += f"{i}. Q: {question}\n   A: {answer}\n"
    return formatted


# -----------------------------------------
# 子进程评分函数（!!!）
# -----------------------------------------
def score_single_ppt(ppt_name, sections_subset, summary_folder, api_key, base_url, model):
    """
    子进程执行：处理 1 个 PPT 的若干未评分 sections（sections_subset）
    返回： (ppt_name, list_of_results_for_these_sections)
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 加载 summary
    summary_file = os.path.join(summary_folder, f"{ppt_name}.json")
    if not os.path.exists(summary_file):
        print(f"⚠️ Summary not found for {ppt_name}")
        return ppt_name, []

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    page_summary_map = {
        s["page"]: s.get("summary", "") for s in summary_data.get("slides", [])
    }

    ppt_result = []

    for section in sections_subset:
        start = section["start"]
        end = section["end"]
        qa_list = section.get("questions", [])
        title = section.get("title", "")

        summary_segment = "\n".join([
            page_summary_map.get(p, "") for p in range(start, end + 1)
        ]).strip()

        if not summary_segment or not qa_list:
            # 如果没有 summary 或没有 QA，则跳过并返回一个空或带标记的条目（这里直接跳过）
            print(f"⚠️ Skip empty segment {title} in {ppt_name}")
            continue

        # ---- 调用评分模型 ----
        formatted = format_qa_pairs(qa_list)
        prompt = score_chapter_prompt.format(
            summary=summary_segment,
            qa_formatted=formatted
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = response.choices[0].message.content.strip()
            score = extract_json(content)

            ppt_result.append({
                "title": title,
                "start": start,
                "end": end,
                "score": score
            })

        except Exception as e:
            print(f"❌ Error scoring {ppt_name} - {title}: {e}")
            # 如果出错可以选择 append 一个标记项，或直接跳过以便下次重试；这里我们跳过（主进程会保留该段未评分）
            continue

    return ppt_name, ppt_result


def process_all_parallel(query_path, summary_folder, output_path,
                         api_key, base_url, model, max_workers=4):
    """
    改进版：粒度按 PPT 内的 title 进行断点续跑
    - 会找出每个 PPT 中尚未评分的 sections（按 title 判断）
    - 只提交这些未评分的 sections 到子进程打分
    - 子任务完成后即时合并并保存
    """

    # --- 加载 query ---
    with open(query_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    # --- 断点续跑：加载已有结果 ---
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            scored_output = json.load(f)
        print(f"🔁 Loaded {len(scored_output)} existing PPT results")
    else:
        scored_output = {}

    # 准备任务列表：对每个 ppt，找出未完成的 sections（按 title）
    tasks = []  # 每个任务是 (ppt_name, sections_subset)
    for ppt_name, sections in queries.items():
        # 已有的该 ppt 的评分条目标题集合
        existing_titles = set()
        if ppt_name in scored_output:
            for item in scored_output[ppt_name]:
                t = item.get("title")
                if t is not None:
                    existing_titles.add(t)

        # 找出缺失的 sections（按 title）
        missing_sections = []
        for sec in sections:
            sec_title = sec.get("title", "")
            if sec_title not in existing_titles:
                missing_sections.append(sec)

        if len(missing_sections) == 0:
            # 全部完成，跳过
            continue

        # 提交一个任务：该 PPT 的 missing sections（粒度为每个 PPT 一次性的若干段）
        tasks.append((ppt_name, missing_sections))

    print(f"📄 Total PPT: {len(queries)}, PPT needing work: {len(tasks)}")

    # --- 启动多进程，按任务并发执行 ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                score_single_ppt,
                ppt_name,
                sections_subset,
                summary_folder,
                api_key,
                base_url,
                model
            ): (ppt_name, sections_subset)
            for (ppt_name, sections_subset) in tasks
        }

        for future in tqdm(as_completed(future_to_task), total=len(future_to_task), desc="Scoring PPTs"):
            ppt_name, _ = future_to_task[future]
            try:
                name, new_results = future.result()
                if name is None:
                    print(f"⚠️ Received empty result for task {ppt_name}")
                    continue

                # 确保 scored_output 中存在 ppt 的 entry（否则初始化）
                if name not in scored_output:
                    scored_output[name] = []

                # 用 title 去重合并新结果（避免重复）
                existing_titles = {item.get("title") for item in scored_output[name]}

                appended = 0
                for r in new_results:
                    t = r.get("title")
                    if t not in existing_titles:
                        scored_output[name].append(r)
                        existing_titles.add(t)
                        appended += 1

                print(f"✅ {name}: appended {appended} new scored sections")

            except Exception as e:
                print(f"❌ Error in process for {ppt_name}: {e}")

            # 每处理 1 个就保存（安全）
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(scored_output, f, indent=2, ensure_ascii=False)

    print(f"✅ 完成！全部结果已保存到 {output_path}")


# -----------------------------------------
# Run
# -----------------------------------------
if __name__ == "__main__":
    dataset = "tutorial"

    query_path = f"./data/{dataset}/query/chapter_queries_ori.json"
    summary_folder = f"./data/{dataset}/summaries_json"
    output_path = f"./data/{dataset}/query/scored_chapter_queries.json"

    api_key = "sk-CJ4dP8IEf6IM9INBy9CVHFoh65xkC5Zd7A0LV5xrGiGGY6Sj"  
    base_url = "https://chatapi.onechats.top/v1"
    model = "gemini-2.5-pro"

    process_all_parallel(
        query_path,
        summary_folder,
        output_path,
        api_key,
        base_url,
        model,
        max_workers=4   # <<<< 根据API速率调整
    )
