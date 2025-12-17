import json
from tqdm import tqdm
# 参数
dataset = "tutorial"
query_path = f"./data/{dataset}/query/chapter_queries_ori.json"
score_path = f"./data/{dataset}/query/scored_chapter_queries.json"
output_path = f"./data/{dataset}/query/chapter_queries.json"


# 加载文件
with open(query_path, "r") as f:
    query_data = json.load(f)

with open(score_path, "r") as f:
    score_data = json.load(f)

# 结果容器
filtered_result = {}

# 遍历每个 PPT
for ppt_name, segments in query_data.items():
    if ppt_name not in score_data:
        print(f"⚠️ No score data for {ppt_name}, skipping.")
        continue

    filtered_queries = []
    segment_scores = score_data[ppt_name]
    score_index = {}
    for s in segment_scores:
        key = (s["start"], s["end"])
        score_index[key] = s

    for i, segment in tqdm(enumerate(segments), desc=f"Processing {ppt_name} segments"):
        # str_i = str(i)
        # if str_i not in segment_scores:
        #     print(f"⚠️ Missing score for segment {i} in {ppt_name}")
        #     continue
        key = (segment["start"], segment["end"])
        score_struct = segment_scores[i]
        if key not in score_index:
            print(f"⚠️ Missing score for segment {i} in {ppt_name}, key={key}")
            continue

        score_struct = score_index[key]
        scores = score_struct["score"]
        if not isinstance(scores, list):
            scores = [scores]
        if scores[0].get("relevance") is None: 
            print(f"⚠️ Segment {i} in {ppt_name} has invalid score format.")
            try:
                scores = list(scores[0].values())
                if scores[0].get("relevance") is None:
                    print(f"❌ Still cannot convert scores to list in {ppt_name} - segment {i}, skipping.")
                    continue
            except Exception as e:
                print(f"❌ Cannot convert scores to list in {ppt_name} - segment {i}: {e}")
                continue  
        # 计算平均分
        for j, score in enumerate(scores):
            try:
                avg = sum([
                    score["relevance"]["score"],
                    score["coverage"]["score"]*0.5,
                    score["clarity"]["score"],
                    score["usefulness"]["score"]
                ]) / 3.5
            except Exception as e:
                print(f"❌ Score format error in {ppt_name} - segment {i}: {e}")
                continue
            print(avg)
            if avg >= 4.5:
                filtered_queries.append({
                    "title": segment["title"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "question": segment["questions"][j]["question"],
                    "answer": segment["questions"][j]["answer"],
                    "score": avg
                    # "answer": 
                })

    if filtered_queries:
        filtered_result[ppt_name] = filtered_queries

# 保存输出
with open(output_path, "w") as f:
    json.dump(filtered_result, f, indent=2)

print(f"✅ 筛选完成，已保存至 {output_path}")
