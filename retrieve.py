import json
import os
from search_engine import SearchEngine, HybridSearchEngine
from tqdm import tqdm
def load_qa_data(qa_path: str) -> list:
    with open(qa_path, "r") as f:
        data = json.load(f)
    return data


from tqdm import tqdm
import math

def compute_hit_at_k(hit_flags):
    return any(hit_flags)

def compute_mrr_at_k(hit_flags):
    for idx, hit in enumerate(hit_flags):
        if hit:
            return 1.0 / (idx + 1)
    return 0.0

def compute_map_at_k(hit_flags):
    ap = 0.0
    hit_count = 0
    for idx, hit in enumerate(hit_flags):
        if hit:
            hit_count += 1
            ap += hit_count / (idx + 1)
    return ap / max(1, sum(hit_flags))

def compute_ndcg_at_k(hit_flags):
    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hit_flags))
    ideal_hits = sorted(hit_flags, reverse=True)
    idcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def extract_page_number(metadata):
    file_name = metadata.get("file_name", "")
    try:
        return int(file_name.split("_")[-1].split(".")[0])
    except Exception:
        return -1

def evaluate_retrieval(qa_list, search_engine, dataset, top_k_list=[1, 3, 5, 10, 20]):
    results = []

    for qa in tqdm(qa_list[dataset], desc="Evaluating Queries"):
        query = qa["question"]
        if "score" in qa.keys() and isinstance(qa["score"], dict):
            if qa["score"]["answerable_without_slides"] == 0:
                continue
        doc_id = dataset
        target_range = set(range(qa["start"], qa["end"] + 1))

        # 检索
        recall = search_engine.search(query)
        nodes = recall["source_nodes"][: max(top_k_list)]

        # 命中分析
        retrieve_result = []
        metadata_list = []
        for node in nodes:
            metadata = node["node"]["metadata"]
            page_num = extract_page_number(metadata)
            retrieve_result.append({
                "page_num": page_num,
                "hit": page_num in target_range,
                "node": node,
            })
            metadata_list.append(page_num in target_range)

        # 多个top-k指标
        metrics = {}
        for k in top_k_list:
            hit_flags = metadata_list[:k]
            metrics[f"hit@{k}"] = compute_hit_at_k(hit_flags)
            metrics[f"mrr@{k}"] = compute_mrr_at_k(hit_flags)
            metrics[f"map@{k}"] = compute_map_at_k(hit_flags)
            metrics[f"ndcg@{k}"] = compute_ndcg_at_k(hit_flags)

        result = {
            "query": query,
            "answer": qa["answer"],
            "doc_id": doc_id,
            "target_range": list(target_range),
            "retrieve_result": retrieve_result,
            **metrics,  # 合并多个top-k指标
        }
        results.append(result)

    return results


def summarize_results(results, top_k_list=[1, 3, 5, 10, 20]):
    total = len(results)
    summary = {"total_num": total}

    print(f"Total Queries: {total}")
    for k in top_k_list:
        hit = sum(r[f"hit@{k}"] for r in results)
        mrr = sum(r[f"mrr@{k}"] for r in results)
        map_total = sum(r[f"map@{k}"] for r in results)
        ndcg_total = sum(r[f"ndcg@{k}"] for r in results)
        summary[f"hit@{k}"] = hit / total
        summary[f"mrr@{k}"] = mrr / total
        summary[f"map@{k}"] = map_total / total
        summary[f"ndcg@{k}"] = ndcg_total / total

        print(f"\nTop-{k} Metrics:")
        print(f"  Hit@{k}: {hit / total:.4f}")
        print(f"  MRR@{k}: {mrr / total:.4f}")
        print(f"  MAP@{k}: {map_total / total:.4f}")
        print(f"  NDCG@{k}: {ndcg_total / total:.4f}")

    return summary

def summarize_overall(all_dataset_summaries, top_k_list=[1, 3, 5, 10, 20]):
    print("\n===============================")
    print("Overall Evaluation Summary:")
    print("===============================")

    # 初始化累积指标
    total_all = 0
    metrics_sum = {f"hit@{k}": 0 for k in top_k_list}
    metrics_sum.update({f"mrr@{k}": 0 for k in top_k_list})
    metrics_sum.update({f"map@{k}": 0 for k in top_k_list})
    metrics_sum.update({f"ndcg@{k}": 0 for k in top_k_list})

    # 加权求和
    for dataset_name, summary in all_dataset_summaries.items():
        total_num = summary["total_num"]
        total_all += total_num
        for k in top_k_list:
            metrics_sum[f"hit@{k}"] += summary[f"hit@{k}"] * total_num
            metrics_sum[f"mrr@{k}"] += summary[f"mrr@{k}"] * total_num
            metrics_sum[f"map@{k}"] += summary[f"map@{k}"] * total_num
            metrics_sum[f"ndcg@{k}"] += summary[f"ndcg@{k}"] * total_num

    # 输出结果
    print(f"Total Queries (All Datasets): {total_all}\n")
    for k in top_k_list:
        hit_avg = metrics_sum[f"hit@{k}"] / total_all
        mrr_avg = metrics_sum[f"mrr@{k}"] / total_all
        map_avg = metrics_sum[f"map@{k}"] / total_all
        ndcg_avg = metrics_sum[f"ndcg@{k}"] / total_all

        print(f"Top-{k} Overall:")
        print(f"  Hit@{k}:  {hit_avg:.4f}")
        print(f"  MRR@{k}:  {mrr_avg:.4f}")
        print(f"  MAP@{k}:  {map_avg:.4f}")
        print(f"  NDCG@{k}: {ndcg_avg:.4f}\n")

    print("========= Final Weighted Summary =========")
    print(f"Datasets Combined: {len(all_dataset_summaries)}")
    print(f"Weighted by number of queries in each dataset.")

if __name__ == "__main__":
    

    # 加载 QA 数据
    # dataset_name = "bain_report_southeast_asias_green_economy_2025"
    dataset_type = "tutorial"
    embed_model_name = "vidore/colpali-v1.2"
    qa_config = "detail"
    qa_name = f"{qa_config}_queries.json"
    # qa_config = qa_name.split("queries.json")[0]
    # topk = 10
    top_k_list=[1, 3, 5, 10, 20]
    retrieve_result_path = os.path.join("./result", dataset_type, f"{qa_config}_retrieve.json")
    all_dataset_summaries = {}
    retrieve_result = {}
    dataset_dir = os.path.join('./data', dataset_type)
    qa_path = os.path.join(dataset_dir, "query", qa_name)
    qa_data = load_qa_data(qa_path)
    if os.path.exists(retrieve_result_path): 
        with open(retrieve_result_path, "r", encoding="utf-8") as f:
            try:
                retrieve_result = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 输出文件损坏，重新开始。")
                retrieve_result = {}
     
    for dataset_name, dataset_results in retrieve_result.items():
        reconstructed_eval_results = []
        for r in dataset_results:
            query = r["query"]
            answer = r.get("answer", "")
            doc_id = r["doc_id"]
            recall_ids = r.get("recall_id", [])

            # 根据 QA 的 target_range 来计算 hit 等指标
            # 这里需要加载 qa_data[dataset_name] 找到对应 query 的 start/end
            qa = next((q for q in qa_data[dataset_name] if q["question"] == query), None)
            if qa is None:
                target_range = set()
            else:
                target_range = set(range(qa["start"], qa["end"] + 1))

            # 构造 retrieve_result 结构
            retrieve_result_list = [{"page_num": pid, "hit": pid in target_range} for pid in recall_ids]
            metadata_list = [item["hit"] for item in retrieve_result_list]

            # 计算 top-k 指标
            metrics = {}
            for k in top_k_list:
                flags = metadata_list[:k]
                metrics[f"hit@{k}"] = compute_hit_at_k(flags)
                metrics[f"mrr@{k}"] = compute_mrr_at_k(flags)
                metrics[f"map@{k}"] = compute_map_at_k(flags)
                metrics[f"ndcg@{k}"] = compute_ndcg_at_k(flags)

            reconstructed_eval_results.append({
                "query": query,
                "answer": answer,
                "doc_id": doc_id,
                "retrieve_result": retrieve_result_list,
                **metrics
            })

        all_dataset_summaries[dataset_name] = summarize_results(reconstructed_eval_results)
    # 初始化搜索引擎

    for dataset_name in tqdm(qa_data.keys(), desc="Initializing Search Engine for Datasets"):

        if dataset_name in retrieve_result.keys():
            print(f"Dataset '{dataset_name}' already evaluated. Skipping.")
            continue
        if dataset_name not in (os.listdir(os.path.join(dataset_dir, "images"))):
            print(f"Dataset '{dataset_name}' not found in {dataset_dir}. Skipping.")
            continue
        search_engine = SearchEngine(
        dataset = dataset_dir,
        dataset_name = dataset_name,
        # node_dir_prefix=dataset_name + "_img_emb",
        embed_model_name=embed_model_name,
        ) 

        # 评估检索效果
        eval_results = evaluate_retrieval(qa_data, search_engine, dataset=dataset_name)
        search_engine.close()
        
        # 输出每个查询的结果
        # for res in eval_results:
        #     print(f"Query: {res['query']}")
        #     print(f"Answer: {res['answer']}")
        #     print(f"Doc ID: {res['doc_id']}")
        #     print(f"Target Range: {res['target_range']}")
        #     print(f"Hit@k: {res['hit@k']}, MRR@k: {res['mrr@k']:.3f}, MAP@k: {res['map@k']:.3f}, NDCG@k: {res['ndcg@k']:.3f}")
        #     print("-" * 50)

        # 汇总整体结果
        retrieve_result[dataset_name] = [{
            "query": eval_result["query"],
            "answer": eval_result["answer"],
            "doc_id": eval_result["doc_id"],
            "recall_id": [i["page_num"] for i in eval_result["retrieve_result"]]
        }
        for eval_result in eval_results
        ]
            

        all_dataset_summaries[dataset_name] = summarize_results(eval_results)
        print(f"Summary for dataset '{dataset_name}': {all_dataset_summaries[dataset_name]}")
        with open(retrieve_result_path, "w", encoding="utf-8") as f:
            json.dump(retrieve_result, f, indent=2)

    summarize_overall(all_dataset_summaries)