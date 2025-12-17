import os
import json
from PIL import Image
from tqdm import tqdm
# from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from typing import List, Union


# === 设置路径和参数 ===
root_dir = "./trend_images"
output_json = "./all_ppt_summaries.json"
prompt_template = "<img>\n请你总结这页幻灯片的主要内容，用简洁中文回答："

# === 初始化 VLLM 模型 ===
model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
class Qwen_VL_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        # vLLM 初始化（多卡并行）
        self.llm = LLM(
            model=model_name,
            dtype="half",
            tensor_parallel_size=4,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=True,
        )
        
        # Transformers 处理器（单卡）
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(model_name)

        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=8192,
            stop_token_ids=[],
        )

    
    def generate(self, query: str, images: List[Union[str, Image.Image]]):
        """
        输入:
            - query: 文本提示（str）
            - images: 图片路径列表或 PIL.Image 列表
        输出:
            - 生成的文本回答
        """
        # 1. 处理图像和文本输入（仍用 Transformers）
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
            
        # 将图片转换为 PIL.Image（如果是路径）
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img)
            processed_images.append(img)
        
        # 2. 构建多模态输入（格式需匹配 Qwen-VL 的模板）
        content = [
            {"type": "image", "image": img} for img in processed_images
        ]
        content.append({"type": "text", "text": query})
        
        messages = [{"role": "user", "content": content}]
        
        # 3. 使用 Processor 处理输入
        prompt = self.processor.apply_chat_template(
            messages, 
            tokenize=False,  # 不 tokenize（vLLM 会处理）
            add_generation_prompt=True
        )
        
        # 4. 提取视觉特征（仍需用 Transformers）
        
        mm_data = {}
        if processed_images is not None:
            mm_data["image"] = processed_images
        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        # 5. 使用 vLLM 生成文本
        outputs = self.llm.generate(
            prompts=[llm_inputs],
            sampling_params=self.sampling_params
        )
        
        # 6. 解码输出
        generated_text = outputs[0].outputs[0].text
        return generated_text


class SlideSummarizer:
    def __init__(self, model: Qwen_VL_2_5, root_dir: str = "./data/slideshare/images", output_dir: str = "./data/slideshare/summaries_json"):
        self.model = model
        self.root_dir = root_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def summarize_all(self):
        for ppt_name in os.listdir(self.root_dir):
            ppt_path = os.path.join(self.root_dir, ppt_name)
            if not os.path.isdir(ppt_path):
                continue

            slides = sorted([
                f for f in os.listdir(ppt_path)
                if f.lower().endswith(".jpg")
            ])
            
            result = {
                "ppt_name": ppt_name,
                "slides": []
            }

            for slide_file in tqdm(slides):
                slide_path = os.path.join(ppt_path, slide_file)

                try:
                    summary = self.model.generate(
                        query="""
                        You are summarizing a slide from a business presentation for the purpose of generating high-quality question-answer pairs later.

                        Please write a detailed summary in English that captures the main points, data, and implications of this slide. 
                        The summary should be informative and self-contained, even without the original image.
                        Do not mention “this slide,” “the slide,” or anything about the layout or presentation.
                        Begin directly with the core facts, insights, or arguments.

                        ### Important constraints:
                        - Do not mention words like "this slide", "this graph", "this image", "this chart", or any visual aspect.
                        - Begin directly with the key information, insight, or conclusion.
                        - Language: English
                        - Length: Around 20–80 words
                        - Focus on business insights, key trends, statistics, and entities if mentioned
                        - Avoid superficial description of layout or design elements
                        """
                        ,
                        images=[slide_path]
                    )
                except Exception as e:
                    summary = f"[ERROR] {str(e)}"

                result["slides"].append({
                    "page": self.extract_page_number(slide_file),
                    "filename": slide_file,
                    "summary": summary.strip()
                })

            output_path = os.path.join(self.output_dir, f"{ppt_name}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"[✓] Finished summarizing: {ppt_name}")

    @staticmethod
    def extract_page_number(filename: str) -> int:
        try:
            return int(filename.split("_")[-1].split(".")[0])
        except Exception:
            return -1  # fallback if parsing fails



model = Qwen_VL_2_5()
summarizer = SlideSummarizer(model)
summarizer.summarize_all()
