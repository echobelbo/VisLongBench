from pdf2image import convert_from_path
from tqdm import tqdm
import os

input_folder = "./data/slideshare/pdf"
output_root = "./data/slideshare/images"
os.makedirs(output_root, exist_ok=True)

from pathlib import Path
import shutil

pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

for pdf_file in tqdm(pdf_files, desc="转换PDF"):
    pdf_path = os.path.join(input_folder, pdf_file)
    pdf_name = Path(pdf_file).stem
    output_dir = os.path.join(output_root, pdf_name)
    
    if os.path.exists(output_dir):
        continue  # 已经转换过，跳过
    os.makedirs(output_dir, exist_ok=True)

    try:
        images = convert_from_path(pdf_path, dpi=200)
        for i, img in enumerate(images):
            img.save(os.path.join(output_dir, f"page_{i+1}.jpg"), "JPEG")
    except Exception as e:
        print(f"[!] 处理 {pdf_file} 时出错：{e}")
