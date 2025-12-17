import json
import os
import base64
import sys
import re
import io
from tqdm import tqdm
import math
from PIL import Image
from openai import OpenAI
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompt.summary_prompt import summary_prompt, all_summary_prompt
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import tempfile



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
def encode_image_to_base64(img_path):
    if isinstance(img_path, Image.Image):
        buffered = io.BytesIO()
        img_path.save(buffered, format="JPEG", quality=90)
        img_data = buffered.getvalue()
        return base64.b64encode(img_data).decode("utf-8")
    elif isinstance(img_path, str) and os.path.exists(img_path):
        # 如果是路径，读取文件内容
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
def concat_images_with_resize(image_paths: List[str], max_size: int = 4096) -> List[Image.Image]:
    """
    改进版：
    - 输入：任意数量图片路径
    - 输出：多张拼接图（例如 20 张 → 返回 4 张）
    - 每张拼接图尽量减少压缩损失
    """

    images = [Image.open(p).convert("RGB") for p in image_paths if os.path.exists(p)]
    if not images:
        raise ValueError("❌ 没有有效图片可拼接。")

    n = len(images)

    # 自动决定分组（每组最多 5 张）
    group_size = 5
    groups = [images[i:i + group_size] for i in range(0, n, group_size)]

    merged_results = []

    for batch in groups:
        merged_results.append(_merge_small_group(batch, max_size))

    return merged_results


def _merge_small_group(images: List[Image.Image], max_size: int) -> Image.Image:
    """
    将小批（≤5 张）拼接成 1 张图，尽量减少损失。
    使用 3×2 网格布局保证分辨率最高。
    """

    n = len(images)
    cols, rows = 3, 2  # 为 <=5 张图片设计的固定布局

    # 统一缩略图大小（不拉伸，只缩小）
    min_w = min(im.width for im in images)
    min_h = min(im.height for im in images)

    thumbs = []
    for im in images:
        im = im.copy()
        im.thumbnail((min_w, min_h), Image.LANCZOS)
        thumbs.append(im)

    # 创建网格拼接画布
    combined = Image.new("RGB", (cols * min_w, rows * min_h), (255, 255, 255))

    for idx, im in enumerate(thumbs):
        r, c = divmod(idx, cols)
        combined.paste(im, (c * min_w, r * min_h))

    # 最终保真缩放
    if max(combined.size) > max_size:
        combined.thumbnail((max_size, max_size), Image.LANCZOS)

    return combined

class PPTQueryGenerator:
    def __init__(self, api_key, base_url="https://chatapi.onechats.top/v1", model="Qwen/Qwen2.5-VL-72B-Instruct"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def build_answer_messages(self, images_b64: List[str], prompt: str):
        """
        构建多模态消息结构
        使用 file_id 上传方式避免 GPT token 爆炸
        """
        messages = [
            {"role": "system", "content": "You are an AI that generates answers for a given PPT image set."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        user_content = messages[1]["content"]

        for img_b64 in images_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}
            })

        return messages

    def build_chapter_sum_messages(self, images_b64, prompt, title):
        """构建多模态消息结构"""
        messages = [
            {
                "role": "system",
                "content": "You are an AI that generates detailed summaries for a given PPT image set."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt.format(title=title)}]
            }
        ]

        for img_b64 in images_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        return messages
    def build_score_answers_messages(self, query, answer, answer_gen, prompt):
        """构建多模态消息结构"""
        messages = [
            {
                "role": "system",
                "content": "You are an AI that evaluates answers for a given PPT image set."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt.format(query=query, answer=answer, answer_gen=answer_gen)}]
            }
        ]
        return messages

    def build_all_sum_messages(self, chunk_summaries, ppt_name):
        """构建多模态消息结构"""
        merge_prompt = all_summary_prompt.format(ppt_name=ppt_name, chapter_summaries_joined="\n\n".join(chunk_summaries))
        messages = [
            {
                "role": "system",
                "content": "You are an AI that generates detailed summaries for a given PPT summary."
            },
            {
                "role": "user",
                "content": merge_prompt
            }
        ]
        return messages
    def get_topk_images(self, doc_id: str, recall_ids: List[int], top_k: int, base_dir: str, concat_images:bool = True):
        """
        根据 recall_id 加载 top_k 图像并拼接为单张图（自动压缩），输出 base64。
        """
        if concat_images:
            folder = os.path.join(base_dir, doc_id)
            selected_ids = recall_ids[:top_k]
            image_paths = [os.path.join(folder, f"page_{rid}.jpg") for rid in selected_ids]

            valid_images = [p for p in image_paths if os.path.exists(p)]
            if not valid_images:
                print(f"⚠️ No valid images found in {folder}")
                return []
            combined_img = concat_images_with_resize(valid_images)
            return [encode_image_to_base64(combined_img)]
        
        else:
            folder = os.path.join(base_dir, doc_id) 
            selected_ids = recall_ids[:top_k] 
            images = [] 
            for rid in selected_ids: 
                img_path = os.path.join(folder, f"page_{rid}.jpg") 
                if os.path.exists(img_path): 
                    images.append(encode_image_to_base64(img_path)) 
                else: 
                    print(f"⚠️ Missing image: {img_path}") 
            return images
   
    def generate_query(self, messages, max_tokens=1500, retry=3):
        """调用模型生成文本，带简单重试机制"""
        for _ in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ API 调用失败：{e}，重试中...")
                err_str = str(e)
                if "quota" in err_str.lower() or "not enough" in err_str.lower() or "insufficient" in err_str.lower():
                    print("检测到额度不足，程序终止。")
                # 方式一：直接退出
                    import sys
                    sys.exit(1)
                elif "inappropriate content" in err_str.lower():
                    print("Law limit....")
                    return "law limit Nothing summary"
                elif "timed out" in err_str.lower():
                    return "cant summay to long"
        return None
    def process_ppt_segments(self, ppt_targets, ppt_image_root, prompt, output_path,
                             chunk_size=10, max_workers=4):
        """
        使用线程池并发处理每个 PPT 的 segment
        - 支持断点续跑
        - 支持实时写文件
        - 可以直接使用 self.client
        """

        # 1️⃣ 加载断点续跑结果
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 输出文件损坏，重新开始。")
                results = {}
        else:
            results = {}

        # 2️⃣ 创建线程池
        tasks = []
        future_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:

            # 为每个 PPT 的每个 segment 提交任务
            for ppt_name, segments in tqdm(ppt_targets.items(), desc="PPTs"):
                ppt_dir = os.path.join(ppt_image_root, ppt_name)
                if not os.path.exists(ppt_dir):
                    print(f"❌ 找不到目录: {ppt_dir}")
                    continue

                if ppt_name not in results:
                    results[ppt_name] = []

                done_ids = {s["group_id"] for s in results[ppt_name]}

                for group_id, seg in enumerate(segments):
                    if group_id in done_ids:
                        continue

                    # 提交线程任务
                    future = pool.submit(
                        self._process_single_segment,
                        ppt_name, ppt_dir, group_id, seg, prompt, chunk_size
                    )
                    future_map[future] = ppt_name
                    tasks.append(future)

            # 3️⃣ 等待完成并实时写入
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                ppt_name = future_map[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"❌ Segment 生成失败: {ppt_name}, {e}")
                    continue

                if result is None:
                    continue

                results[ppt_name].append(result)

                # 实时写入
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        return results

    # -------------------------
    # 线程池调用的单个 segment 处理
    # -------------------------
    def _process_single_segment(self, ppt_name, ppt_dir, group_id, seg, prompt, chunk_size):
        """
        处理单个 segment（同步函数）
        可以直接使用 self.client / self.generate_query / self.build_chapter_sum_messages
        """

        start_page = seg["start"]
        end_page = seg["end"]
        title = seg["title"]

        # 跳过前言
        if "foreword" in title.lower():
            print(f"⚠️ 跳过前言部分：{ppt_name} - {title}")
            return None

        # 收集图片
        all_images = []
        for page in range(start_page, end_page + 1):
            img_path = os.path.join(ppt_dir, f"page_{page}.jpg")
            if os.path.exists(img_path):
                all_images.append(encode_image_to_base64(img_path))
            else:
                print(f"⚠️ 找不到图片: {img_path}")

        if not all_images:
            return None

        # 分块处理
        chunks = [all_images[i:i+chunk_size] for i in range(0, len(all_images), chunk_size)]
        chunk_summaries = []

        for chunk_images in chunks:
            messages = self.build_chapter_sum_messages(chunk_images, prompt, title)
            summary = self.generate_query(messages)
            chunk_summaries.append(summary)

        # 合并 chunk 总结
        if len(chunk_summaries) == 1:
            merged_summary = chunk_summaries[0]
        else:
            merge_prompt = (
                f"You are an expert summarizer.\n"
                f"Below are several detailed summaries from different chunks of the same presentation section titled '{title}'. "
                f"Please integrate them into one coherent and comprehensive summary.\n\n"
                f"Chunk summaries:\n" + "\n\n".join(chunk_summaries)
            )
            merged_summary = self.generate_query([{"role": "user", "content": merge_prompt}])

        # 返回 segment 结果
        return {
            "group_id": group_id,
            "title": title,
            "start": start_page,
            "end": end_page,
            "summary": merged_summary
        }

    def summary_all_chapters(self, chapter_summaries, ppt_name):
        messages = self.build_all_sum_messages(chapter_summaries, ppt_name)
        final_summary = self.generate_query(messages)
        return final_summary