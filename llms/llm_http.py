import torch
from PIL import Image
from pathlib import Path
import sys
import base64
from io import BytesIO
import os
from vllm import LLM as VLLM, SamplingParams
from typing import List, Union
import requests
# from scripts.script_prompt import en_prompt
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
API_URL = "http://localhost:8000/v1/chat/completions"

def _encode_image(image_path: Union[str, Image.Image]) -> str:
    """统一处理图片编码"""
    if isinstance(image_path, Image.Image):
        buffered = BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
    else:
        with open(image_path, "rb") as f:
            img_data = f.read()
    return base64.b64encode(img_data).decode('utf-8')

class QwenVLvLLM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"):
        # vLLM初始化
        self.model_name = model_name
        # self.llm = VLLM(
        #     model=model_name,
        #     tensor_parallel_size=4,
        #     # gpu_memory_utilization=0.9,
        #     max_model_len=32768,
        #     trust_remote_code=True,
        #     dtype="half"
        # )
        
        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            stop=["<|im_end|>"]
        )

    def generate(self, query: str, images: List[str], is_json) -> str:
        # 构建多模态输入
        image_url = []
        for img in images:
            base64_image = _encode_image(img)
            image_url.append(
                f"data:image/jpeg;base64,{base64_image}"
            )
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image_url", "image_url": {"url":img}} for img in image_url],
                {"type": "text", "text": query}
            ]
        }]
        
        mess = {
            "model" : self.model_name,
            "messages" : messages,
            "max_tokens" :2048,
            # "response_format": {"type": "json_object"}
        }
        if is_json:
            mess["response_format"] = {"type": "json_object"}
        response = requests.post(
            API_URL,
            json=mess,   
        )
        return response.json()["choices"][0]["message"]["content"]

class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        
        if 'Qwen2.5-VL' in model_name:
            self.model = QwenVLvLLM(model_name)
        elif model_name.startswith('gpt'):
            from openai import OpenAI
            self.client = OpenAI()

    def generate(self, 
                query: str = "", 
                image: List[str] = [],
                is_json: bool = False,
                **kwargs) -> str:
        if 'Qwen2.5-VL' in self.model_name:
            return self.model.generate(query, image, is_json)
        elif self.model_name.startswith('gpt'):
            pass
            # content = [{"type": "text", "text": query}]
            # for img in images:
            #     base64_image = _encode_image(img)
            #     content.append({
            #         "type": "image_url",
            #         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            #     })
            # response = self.client.chat.completions.create(
            #     model=self.model_name,
            #     messages=[{"role": "user", "content": content}],
            #     **kwargs
            # )
            # return response.choices[0].message.content

if __name__ == '__main__':
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    API_URL = "http://localhost:8000/v1/chat/completions"
    # Qwen多模态测试
    vl_llm = LLM("Qwen/Qwen2.5-VL-7B-Instruct")
    entity_dict = vl_llm.generate(
        query="how many pictures can you see?",
        images=["/data2/home/yankai/ppt_crawler/data/slideshare/images/2736f8bd-1743-4c53-aed0-26b0ec6908da/page_3.jpg",
                "/data2/home/yankai/ppt_crawler/data/slideshare/images/2736f8bd-1743-4c53-aed0-26b0ec6908da/page_2.jpg",
                #  "/data2/home/yankai/ppt_crawler/data/slideshare/images/2736f8bd-1743-4c53-aed0-26b0ec6908da/page_5.jpg",
                  "/data2/home/yankai/ppt_crawler/data/slideshare/images/2736f8bd-1743-4c53-aed0-26b0ec6908da/page_23.jpg" ],
        is_json=True
    )
    print(type(entity_dict))
    print(entity_dict)
    # relation_dict = vl_llm.generate(
    #     query=en_prompt["relation_extract"].format(entity = entity_dict),
    #     images=["/data2/home/yankai/VisGRAG/data_test/img/0903organizingforbiandbigdatainthe21stcentury-clean-140922112024-phpapp02_95_11.jpg"],
    #     is_json=True
        
    # )
    # print(relation_dict)
    # print(vl_llm.generate(
    #     query="discribe the picture",
    #     images=["/data2/home/yankai/VisGRAG/data_test/img/0903organizingforbiandbigdatainthe21stcentury-clean-140922112024-phpapp02_95_10.jpg"],
    #     is_json=False
    # ))
    # # GPT-4o测试
    # gpt_llm = LLM('gpt-4o')
    # print(gpt_llm.generate(
    #     query="用三个词描述这张图片",
    #     images=["path/to/image.jpg"]
    # ))