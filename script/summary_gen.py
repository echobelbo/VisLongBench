import json
import os
import base64
import sys
import re
from tqdm import tqdm
from openai import OpenAI
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt.summary_prompt import summary_prompt, all_summary_prompt
from llms.ppt_generater import PPTQueryGenerator
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



# === 主函数 ===
def main():
    dataset = "tutorial"
    json_path = f"./data/{dataset}/chunk.json"
    model_name = "gemini-2.5-pro" ### [qwen2.5-vl-7b-instruct,gpt-4o, gemini-2.5-pro, claude-3-5-sonnet-latest, gpt-4o-mini, Qwen/Qwen2.5-VL-72B-Instruct]

    ppt_image_root = f"./data/{dataset}/images"
    safe_name=model_name.replace("/", "_")
    chapter_output_path = f"./data/{dataset}/summary_llm/{safe_name}_summary.json"
    all_summary_output_path = f"./data/{dataset}/summary_llm/{safe_name}_final_summary.json"



    # api_key = "sk-zoQjwxjRKPK0sCONvPB9HZNhqjajeM8ZgYacD5dz5mu5f77U"#deepseek
    
    api_key = "sk-xn3D5pK91opNGieQbWMkBcMLl4lDUBzMwbaz4FjioxmwVEvC" #gemini
    # api_key = "sk-CJ4dP8IEf6IM9INBy9CVHFoh65xkC5Zd7A0LV5xrGiGGY6Sj" #default
    with open(json_path, "r", encoding="utf-8") as f:
        ppt_targets = json.load(f)
    
    # model_name="qwen2.5-vl-7b-instruct"
    if model_name == "qwen2.5-vl-7b-instruct":
        api_key="sk-464323d0d0184c5aa82659a7b95663da"
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        generator = PPTQueryGenerator(api_key=api_key, base_url=base_url,model=model_name)
    else:
        generator = PPTQueryGenerator(api_key=api_key, model=model_name)
    results = generator.process_ppt_segments(ppt_targets, ppt_image_root, summary_prompt, chapter_output_path)
    final_summaries = {}
    if os.path.exists(all_summary_output_path):
            with open(all_summary_output_path, "r", encoding="utf-8") as f:
                try:
                    final_summaries = json.load(f)
                except json.JSONDecodeError:
                    print("⚠️ 输出文件损坏，重新开始。")
                    final_summaries = {}
    else:
            final_summaries = {}
    for ppt_name, segments in tqdm(results.items(), desc="Generating Final Summaries", unit="ppt"):
        # print(f"PPT: {ppt_name}")
        chapter_summaries = []
        for seg in segments:
            chapter_summaries.append(seg["summary"])
        if ppt_name in final_summaries:
            print(f"⏩ 跳过已完成的最终总结：{ppt_name}")
            continue
        final_summaries[ppt_name] = generator.summary_all_chapters(chapter_summaries, ppt_name)

        with open(all_summary_output_path, "w", encoding="utf-8") as f:
            json.dump(final_summaries, f, ensure_ascii=False, indent=2)
    print(f"✅ 所有总结已生成并写入 {all_summary_output_path}")



if __name__ == "__main__":
    main()
