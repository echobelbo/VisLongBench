import torch
from PIL import Image
from pathlib import Path
import sys
import base64
from io import BytesIO
import os

from vllm import LLM, SamplingParams
from typing import List, Union

def _encode_image(image_path):
    if isinstance(image_path,Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        base64_encoded = base64.b64encode(img_data).decode("utf-8")
        return base64_encoded
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
class Qwen_Text:
    def __init__(self, model_name):
        self.llm = LLM(
            model=model_name,
            dtype="half",
            tensor_parallel_size=4,
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=8192,
            stop_token_ids=[],
        )
    def _build_prompt(self, query: str, system_prompt: str = "You are an expert in entity and relation extraction from text") -> str:
        """
        构造 Qwen2.5 聊天模型的 prompt 格式
        """
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query.strip()}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def generate(self, query: str, system_prompt: str = "You are an expert in entity and relation extraction from text") -> str:
        prompt = self._build_prompt(query, system_prompt)

        # 调用 vLLM 推理
        outputs = self.llm.generate(
            prompts=[prompt],
            sampling_params=self.sampling_params
        )

        return outputs[0].outputs[0].text.strip()
        

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
            tokenize=False, 
            add_generation_prompt=True
        )
        
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
    
        
class mLLM:
    def __init__(self,model_name):
        self.model_name =model_name
        if 'Qwen2.5-VL' in self.model_name:
            self.model = Qwen_VL_2_5(model_name)
        elif "Qwen2.5-7B-Instruct" in self.model_name:
            self.model = Qwen_Text(model_name)
        elif model_name.startswith('gpt'):
            from openai import OpenAI
            self.model = OpenAI(api_key="sk-WZO2Fg9D6zpipJmlKKFINkbFWR4iAcfYu43dgVDEtPYLs2Fe", base_url="https://chatapi.onechats.top/v1")
            
    def generate(self,**kwargs):
        query = kwargs.get('query','')
        image = kwargs.get('image','')
        model_name = kwargs.get('model_name','')

        if 'Qwen2.5' in self.model_name:
            return self.model.generate(query,image)
        elif self.model_name.startswith('gpt'):
            content = [{
                "type": "text",
                "text": query
            }]
            if image != '':
                filepaths = [Path(img).resolve().as_posix() for img in image]
                for filepath in filepaths:
                    base64_image = _encode_image(filepath)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}}
                        )
            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": "user",
                    "content": content
                    }

                ],
                max_tokens=1024,
                )
            return completion.choices[0].message.content

if __name__ == '__main__':
    llm = LLM('Qwen/Qwen2.5-VL-7B-Instruct')
    response = llm.generate(query='how many pictures can you see?',image=['image_path'])
    print(response)