import os
import json
import base64
from tqdm import tqdm
import time
import random
from PIL import Image
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================
# 工具函数：图像转 base64
# ============================================
def encode_image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ============================================
# 多模态调用 Qwen2.5-VL 的标准结构
# ============================================
class Qwen_VL_API:
    def __init__(self,  api_key="sk-f1f8941557654a2a932a15f555b29d7d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model_name="qwen2.5-vl-7b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_name

    def build_messages(self, images_b64, prompt):
        """
        构建多模态输入（严格符合你给的参考格式）
        """
        messages = [
            {
                "role": "system",
                "content": "You are an AI helping summarize PPT slides."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]

        # 将图片附加到 user role 的 content 中
        for img_b64 in images_b64:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }
            })

        return messages

    def generate(self, prompt, image_paths):
        """
        prompt: 文本提示
        image_paths: [img1, img2, ...]
        """
        images_b64 = [encode_image_to_base64(p) for p in image_paths]
        messages = self.build_messages(images_b64, prompt)

        resp = self.safe_generate(self.client, self.model, messages)

        return resp.strip()
    
    def safe_generate(self, client: OpenAI, model: str, messages, max_retries=3):
        """
        包含 429 + 网络异常 + 随机退避 的最安全 API 调用
        """
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                return resp.choices[0].message.content

            except Exception as e:
                err = str(e)

                # ---- 429 速率限制 ----
                if "429" in err or "rate" in err.lower():
                    wait = 2 ** attempt + random.uniform(0, 0.5)
                    print(f"⚠️ 429 Rate Limit, waiting {wait:.1f}s then retry...")
                    time.sleep(wait)
                    continue

                # ---- 503 / 502 / 网络抖动 ----
                if "503" in err or "502" in err or "timeout" in err.lower():
                    wait = 1.5 ** attempt + 0.1
                    print(f"⚠️ Server unavailable, retry in {wait:.1f}s...")
                    time.sleep(wait)
                    continue

                # ---- 额度不足 / 其他不可恢复 ----
                if "quota" in err.lower() or "insufficient" in err.lower():
                    raise RuntimeError("❌ API 额度不足，程序终止")

                # 未知异常 → 直接抛出
                raise e

        raise RuntimeError("❌ Reached max retries, still can't get result")

# ============================================
# 幻灯片总结器：遍历 PPT 文件夹并总结所有页面
# ============================================
def summarize_worker(slide_path, slide_file, prompt):
    # 子进程里初始化 API 客户端（例如硅基流动）
    model = Qwen_VL_API()

    try:
        summary = model.generate(
            prompt=prompt,
            image_paths=[slide_path]
        )
    except Exception as e:
        summary = ""
        print(e)

    return slide_file, summary



class SlideSummarizer:
    def __init__(self,
                 root_dir="./data/slideshare/images",
                 output_dir="./data/slideshare/summaries_json"):

        # self.model_name = model
        self.root_dir = root_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.slide_prompt = """
You are summarizing a page from a presentation for the purpose of generating high-quality question–answer pairs later.

Write a clear and self-contained summary in English that captures the main ideas, data points, relationships, and implications conveyed on the page. 
Base your summary on the content, logical structure, and visual organization, including how information is grouped or emphasized.

- Output must be natural, human-readable English text
- Do NOT include any code, tool calls, placeholders, or special tokens
- Do not mention “slide”, “graph”, “image”, “layout”, or any visual aspects
- Begin directly with key information or insights
- Length: 10–80 words
- Focus on business insights, key trends, statistics, and entities if mentioned
- If the slide has only a few words, summarize them directly
- If the slide has no meaningful content, write "No significant content"
- Avoid superficial description of layout or design elements

Begin directly with key information or insights.
Length: 60–80 words.(If necessary)
"""

    def summarize_all(self, max_workers=4):

        for ppt_name in os.listdir(self.root_dir):
            ppt_path = os.path.join(self.root_dir, ppt_name)
            if not os.path.isdir(ppt_path):
                continue

            print(f"📘 Processing PPT: {ppt_name}")
            slides = sorted([f for f in os.listdir(ppt_path) if f.lower().endswith(".jpg")])

            output_path = os.path.join(self.output_dir, f"{ppt_name}.json")

            # ===== 断点加载 =====
            if os.path.exists(output_path):
                result = json.load(open(output_path, "r", encoding="utf-8"))
                done = {x["filename"] for x in result["slides"]}
            else:
                result = {"ppt_name": ppt_name, "slides": []}
                done = set()

            tasks = [
                (os.path.join(ppt_path, slide_file), slide_file, self.slide_prompt)
                for slide_file in slides
                if slide_file not in done
            ]

            if not tasks:
                print(f"✓ Already completed {ppt_name}")
                continue

            print(f"🚀 Dispatch {len(tasks)} tasks with {max_workers} workers")

            # ============= 多进程池 =============
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(summarize_worker, *t)
                    for t in tasks
                ]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Sum of {ppt_name}"):
                    slide_file, summary = fut.result()

                    result["slides"].append({
                        "page": self.extract_page_number(slide_file),
                        "filename": slide_file,
                        "summary": summary
                    })

                    json.dump(result, open(output_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


    @staticmethod
    def extract_page_number(filename):
        try:
            return int(filename.split("_")[-1].split(".")[0])
        except:
            return -1

def clean_error_summaries(json_root):
    """
    清洗所有包含 [ERROR] 的 summary，删除或清空 summary 字段
    """
    for file in os.listdir(json_root):
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(json_root, file)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        changed = False
        cleaned_slides = []
        for slide in data.get("slides", []):
            summary = slide.get("summary", "")
            if not summary or (isinstance(summary, str) and (summary.startswith("[ERROR]") or "addCriterion" in summary)):
                changed=True
                continue
            else:            
                cleaned_slides.append(slide)

        data["slides"] = cleaned_slides

        if changed:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✔ Cleaned: {file}")

    print("🎉 All JSON files cleaned.")


# model = Qwen_VL_API(
#     api_key="sk-wexqcpmdjrxbzxftdanjiprskxlrahgzpwakfqopzoxzubyd",
#     base_url="https://api.siliconflow.cn/v1",
#     model_name="Pro/Qwen/Qwen2.5-VL-7B-Instruct"
# )
clean_error_summaries("./data/slideshare/summaries_json")
summarizer = SlideSummarizer()
summarizer.summarize_all()
