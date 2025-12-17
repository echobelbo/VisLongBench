import json
import os
import base64
import sys
import re
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from llms.ppt_generater import PPTQueryGenerator
from prompt.answer_prompt import chapter_answer, score_answer, detail_answer, direct_answer
from typing import List

def parse_args():
    parser = argparse.ArgumentParser(description="Run PPT QA or retrieval evaluation.")
    parser.add_argument(
        "--query_type",
        type=str,
        default="chapter",
        choices=["direct", "detail", "chapter"],
        help="Type of question generation or evaluation (default: direct)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Top-K value for retrieval evaluation (default: 10)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-vl-7b-instruct",
        help="Model name to use for generation and evaluation (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--test_all_topk",
        action="store_true",
        help="Whether to test all top-K results"
    )
    parser.add_argument(
        "--concat_images",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        default="tutorial",
        choices=["trend", "tutorial", "slideshare"]
    )
    return parser.parse_args()


def load_retrieve_result(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_one_qa(generator, ppt_name, qa, query_type, topk, concat_images):
    """处理单个 QA 样本"""
    query = qa["query"]
    retrieved_images = qa["recall_id"]

    # 取 top-k 图像
    retrieved_images = generator.get_topk_images(
        ppt_name, retrieved_images, topk, f"./data/{dataset}/images", concat_images
    )

    # 构建 prompt
    if query_type == "chapter":
        messages = generator.build_answer_messages(retrieved_images, chapter_answer.format(query=query))
    elif query_type == "detail":
        messages = generator.build_answer_messages(retrieved_images, detail_answer.format(query=query))
    else:
        messages = generator.build_answer_messages(retrieved_images, direct_answer.format(query=query))

    # 调用模型
    answer = generator.generate_query(messages)
    qa["gen_answer"] = answer
    return qa
def generate_answers_concurrent(generator, retrieve_result, query_type, output_path,
                                topk=3, concat_images=True, max_workers=4):

    answers = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
              try:
                  answers = json.load(f)
                  print(f"⏩ 已完成的答案加载：{output_path}")
              except json.JSONDecodeError:
                  print("⚠️ 输出文件损坏，重新开始。")
                  answers = {}
    for ppt_name, qa_list in tqdm(retrieve_result.items(), desc="Processing Answers", unit="ppt"):
        if ppt_name in answers:
            print(f"⏩ 跳过已完成的 PPT：{ppt_name}")
            continue
        answers[ppt_name] = []

        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, qa in enumerate(qa_list):
                qa["id"] = i
                futures.append(
                    executor.submit(
                        process_one_qa,
                        generator, ppt_name, qa, query_type, topk, concat_images
                    )
                )

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Answers of {ppt_name}"):
                try:
                    qa_result = f.result()
                    answers[ppt_name].append(qa_result)
                except Exception as e:
                    print(f"⚠️ Error processing {ppt_name}: {e}")

        # 保存该 PPT 的结果（防止中途崩溃丢失）
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)

    return answers

def generate_answers(generator: PPTQueryGenerator, topk: int, retrieve_result: dict, query_type: str, output_path: str, concat_images:bool = False):
    answers = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
              try:
                  answers = json.load(f)
              except json.JSONDecodeError:
                  print("⚠️ 输出文件损坏，重新开始。")
                  answers = {}
    for ppt_name, qa_list in tqdm(retrieve_result.items(), desc="Processing Answers", unit="ppt"):
        if ppt_name not in answers:
            answers[ppt_name] = []
        else:
            continue  # 已经处理过，跳过
        for i, qa in enumerate(tqdm(qa_list, desc=f"Answers of {ppt_name}", unit="qa")):
            query = qa["query"]
            qa["id"] = i
            # doc_id = qa["doc_id"]
            retrieved_images = qa["recall_id"]
            if query_type == "chapter":
                retrieved_images = generator.get_topk_images(ppt_name, retrieved_images, topk, f"./data/{dataset}/images", concat_images)
                messages = generator.build_answer_messages(retrieved_images, chapter_answer.format(query=query))
            elif query_type == "detail":
                retrieved_images = generator.get_topk_images(ppt_name, retrieved_images, topk, f"./data/{dataset}/images", concat_images)
                messages = generator.build_answer_messages(retrieved_images, detail_answer.format(query=query))
            elif query_type == "direct":
                retrieved_images = generator.get_topk_images(ppt_name, retrieved_images, topk, f"./data/{dataset}/images", concat_images)
                messages = generator.build_answer_messages(retrieved_images, direct_answer.format(query=query))
            answer = generator.generate_query(messages)
            qa["gen_answer"] = answer
            answers[ppt_name].append(qa)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(answers, f, indent=2, ensure_ascii=False)
    return answers

def score_one_answer(generator,  qa):
    """处理单个 QA 样本的评分"""
    query = qa["query"]
    answer = qa["answer"]
    gen_answer = qa["gen_answer"]

    messages = generator.build_score_answers_messages(query, answer, gen_answer, score_answer)
    score = generator.generate_query(messages)
    return {"id": qa["id"], "score": score}

def score_answers_concurrently(generator: PPTQueryGenerator, answers: dict, output_path: str):
    scores = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                scores = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 输出文件损坏，重新开始。")
                scores = {}

    for ppt_name, qa_list in tqdm(answers.items(), desc="Scoring Answers", unit="ppt"):
        if ppt_name in scores.keys():
            continue  # 已经处理过，跳过
        scores[ppt_name] = []

        futures = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for qa in qa_list:
                futures.append(
                    executor.submit(
                        score_one_answer,
                        generator,
                        qa,
                    )
                )

            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Scoring {ppt_name}"):
                try:
                    score = f.result()
                    scores[ppt_name].append(score)
                except Exception as e:
                    print(f"⚠️ Error processing {ppt_name}: {e}")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

    return scores

def score_answers(generator: PPTQueryGenerator, answers: dict, output_path: str):
    scores = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
              try:
                  scores = json.load(f)
              except json.JSONDecodeError:
                  print("⚠️ 输出文件损坏，重新开始。")
                  scores = {}
    for ppt_name, qa_list in tqdm(answers.items(), desc="Scoring Answers", unit="ppt"):
        
        if ppt_name in scores.keys():
            continue  # 已经处理过，跳过
        scores[ppt_name] = []
        for qa in tqdm(qa_list, desc=f"Scoring {ppt_name}", unit="qa"):
            query = qa["query"]
            answer = qa["answer"]
            gen_answer = qa["gen_answer"]
            messages = generator.build_score_answers_messages(query, answer, gen_answer, score_answer)
            score = generator.generate_query(messages)
            scores[ppt_name].append({"id": qa["id"], "score": score})
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
    return scores

def evaluate_score(scores: dict, query_type: str):
    total_score = 0
    total_count = 0
    for ppt_name, score_list in scores.items():
        for item in score_list:
            try:
                # if query_type == "direct"
                score_value = float(re.search(r'\d+(\.\d+)?', item["score"]).group())
                # if query_type == "direct":
                #     score_value = 3 if int(score_value) == 3 else 0 
                total_score += score_value / 3
                total_count += 1
            except:
                continue
    average_score = total_score / total_count if total_count > 0 else 0
    print(f"平均评分: {average_score:.4f}（基于 {total_count} 个回答）")
    return average_score



if __name__ == "__main__":
    api_key = "sk-CJ4dP8IEj"
    api_key_gem = "sk-xn3DwVEvC" #gemini
    api_key_qwen = "sk-f1f15f555b29d7d" #qwen2.5-vl-7b-instruct
    api_key_ds = "sk-zoQjwxjRZgYacD5dz5mu5f77U" #Qwen/Qwen2.5-VL-72B-Instruct
    args = parse_args()  
    query_type = args.query_type
    dataset = args.dataset
    topk = [args.topk]
    gen_model = args.model
    concat_images = args.concat_images
    safe_name=gen_model.replace("/", "_")
    if not os.path.exists(f"./result/{dataset}/{safe_name}"):
        os.makedirs(f"./result/{dataset}/{safe_name}")

    retrieve_result_path = f"./result/{dataset}/{query_type}_retrieve.json"
    if "72B" in gen_model:
        generator = PPTQueryGenerator(api_key=api_key_ds,model=gen_model)
    elif "7b" in gen_model:
        generator = PPTQueryGenerator(api_key=api_key_qwen, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",model=gen_model)
    elif "gemini" in gen_model:
        generator = PPTQueryGenerator(api_key=api_key_gem, model=gen_model)
    else:
        generator = PPTQueryGenerator(api_key=api_key, model=gen_model)
    scorer = PPTQueryGenerator(api_key=api_key, model="gpt-4o")
    retrieve_result = load_retrieve_result(retrieve_result_path)
    if args.test_all_topk:
        topk = [1, 3, 5, 10]
    for k in topk:
        generate_answer_path = f"./result/{dataset}/{safe_name}/{query_type}_topk{k}_answers.json"
        answers = generate_answers_concurrent(generator=generator, topk=k, retrieve_result=retrieve_result, query_type=query_type, output_path=generate_answer_path, concat_images=concat_images)
        # answers = generate_answers(generator, k, retrieve_result, query_type, generate_answer_path, concat_images=concat_images)
        score_output_path = f"./result/{dataset}/{safe_name}/{query_type}_topk{k}_scores.json"
        scores = score_answers_concurrently(scorer, answers, score_output_path)
        # scores = score_answers(scorer, answers, score_output_path)
        evaluate_score(scores, query_type)
        print(f"✅ 所有评分已生成并写入 {score_output_path}")



    