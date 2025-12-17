import json
import os
from tqdm import tqdm
import re
from openai import OpenAI
from prompt.summary_prompt import compare_prompt, score_prompt




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
def evaluate_summaries(json_path_A, json_path_B, output_path, api_key, model="gpt-4o"):
    client = OpenAI(api_key=api_key,base_url="https://chatapi.onechats.top/v1")
    sumA_len = 0
    with open(json_path_A, "r", encoding="utf-8") as f:
        data_A = json.load(f)
    with open(json_path_B, "r", encoding="utf-8") as f:
        data_B = json.load(f)

    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 输出文件损坏，重新开始。")
                results = {}
    else:
        results = {}
    for ppt_name in tqdm(data_A.keys(), desc="Evaluating PPTs"):
        summary_A = data_A[ppt_name]
        summary_B = data_B[ppt_name]
        sumA_len+=len(summary_A)
        if ppt_name not in results:
            results[ppt_name] = {}
        else:
            if "len" not in results[ppt_name]:
                results[ppt_name]["len"] = len(summary_B)
            continue  # 已经评估过，跳过

        # title = data_A[ppt_name]["title"]
        prompt = compare_prompt.format(summary_A = summary_A, summary_B = summary_B)
        # prompt = score_prompt.format(summary_A = summary_A, summary_B = summary_B, title = title)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # max_tokens=800
        )
        try:
            eval_result = extract_json(response.choices[0].message.content)
        except Exception:
            eval_result = {"error": "parse_failed", "raw": response.choices[0].message.content}
        results[ppt_name] = eval_result
        # 实时写入，防止中断丢失
        # eval_result["A_len"] = len(summary_A)
        eval_result["len"] = len(summary_B)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Evaluation completed. Results saved to {output_path}")
    print(f"Average Length of Summary A: {sumA_len / len(data_A):.2f} characters")
    return results
def evaluate_score(summary_score):
    comprehensive_scores = 0
    clarity_scores = 0
    structure_scores = 0
    interpretative_scores = 0
    conciseness_scores = 0
    length = 0
    for ppt_name, evaluations in summary_score.items():
        if "error" in evaluations:
            continue
        scores = evaluations["scores"]
        # overall_judgment = evaluations.get("overall_judgment", "N/A")
        # if overall_judgment == "better":

        comprehensive_scores += scores["comprehensiveness"]
        clarity_scores += scores["clarity"]
        structure_scores += scores["structure"]
        interpretative_scores += scores["interpretative"]
        conciseness_scores += scores["conciseness"]
        length += evaluations["len"]
    num_evaluations = len(summary_score)
    print("Average Scores:")
    print(f"Comprehensiveness: {comprehensive_scores / num_evaluations:.4f}")

    print(f"Clarity: {clarity_scores / num_evaluations:.4f}")
    print(f"Structure: {structure_scores / num_evaluations:.4f}")
    print(f"Interpretative Depth: {interpretative_scores / num_evaluations:.4f}")
    print(f"Conciseness: {conciseness_scores  / num_evaluations:.4f}")
    print(f"Average Length of Summary B: {length / num_evaluations} characters")

    return {
            "comprehensiveness": comprehensive_scores / num_evaluations,
            "clarity": clarity_scores / num_evaluations,
            "structure": structure_scores / num_evaluations,
            "interpretative": interpretative_scores / num_evaluations,
            "conciseness": conciseness_scores / num_evaluations,
            "length": length / num_evaluations
        }

if __name__ == "__main__":
    api_key = ""
    modelA_name = "gpt-4o"
    dataset = "slideshare"

    # modelB_name = "claude-3-5-sonnet-latest"
    eval_output_path = f"./result/{dataset}/sum_eval_result.json"
    eval_results = {}
    for model in ["claude-3-5-sonnet-latest", "Qwen/Qwen2.5-VL-72B-Instruct",  "qwen2.5-vl-7b-instruct","gemini-2.5-pro" ]: ### , "gemini-2.5-pro" 
        safe_model_name = model.replace("/", "_")
        summaryA_path = f"./data/{dataset}/summary_llm/{modelA_name}_final_summary.json"
        summaryB_path = f"./data/{dataset}/summary_llm/{safe_model_name}_final_summary.json"
        score_output_path = f"./data/{dataset}/summary_llm/{safe_model_name}_score.json"
        # eval_output_path = f"./data/trend/summary_llm/{safe_model_name}_eval.json"
        results = evaluate_summaries(summaryA_path, summaryB_path, score_output_path, api_key, "gpt-4o")
        eval_results[safe_model_name] = evaluate_score(results)

    with open(eval_output_path, "w", encoding="utf-8") as f:
       json.dump(eval_results, f, ensure_ascii=False, indent=2)
