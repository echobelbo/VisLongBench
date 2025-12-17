import gradio as gr
from PIL import Image
import os
import json
import re
from collections import Counter
from natsort import natsorted

# 配置
image_root = "./data/trend/images"
output_root = "./data/trend/labels"
port = 7860
progress_file = os.path.join(output_root, "progress.json")

# 标签系统
LABELS = ["chart", "flowchart", "table", "normal", "other"]

def extract_page_number(filename):
    match = re.search(r'(?:slide_)?(\d+)\.(jpg|jpeg|png)$', filename, re.IGNORECASE)
    return int(match.group(1)) if match else 0

def sort_by_page(files):
    return sorted(files, key=lambda x: extract_page_number(os.path.basename(x)))

def save_progress(folder_name, index):
    progress_data = {}
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
    progress_data[folder_name] = index
    with open(progress_file, "w") as f:
        json.dump(progress_data, f, indent=2)

def load_progress(folder_name):
    if not os.path.exists(progress_file):
        return 0
    with open(progress_file, "r") as f:
        progress_data = json.load(f)
        return progress_data.get(folder_name, 0)

def initialize_data():
    image_files = {}
    all_data = {}
    tag_counter = Counter()
    
    for pdf_dir in natsorted(os.listdir(image_root)):
        dir_path = os.path.join(image_root, pdf_dir)
        if not os.path.isdir(dir_path):
            continue
            
        imgs = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        imgs = sort_by_page(imgs)
        image_files[pdf_dir] = imgs
        
        json_path = os.path.join(output_root, f"{pdf_dir}_labels.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                saved_data = json.load(f)
                for item in saved_data:
                    if item.get("label_en"):
                        tag_counter[item["label_en"]] += 1
                all_data[pdf_dir] = saved_data
        else:
            all_data[pdf_dir] = [{
                "filename": f,
                "page_num": extract_page_number(os.path.basename(f)),
                "label_en": None
            } for f in imgs]
    
    return image_files, all_data, tag_counter

image_files, all_data, tag_counter = initialize_data()
os.makedirs(output_root, exist_ok=True)

def handle_labeling(choice, folder_name, index):
    index = int(index)
    folder_data = all_data[folder_name]
    
    if index < len(folder_data):
        folder_data[index]["label_en"] = choice
        tag_counter[choice] += 1
        
        with open(os.path.join(output_root, f"{folder_name}_labels.json"), "w") as f:
            json.dump(folder_data, f, indent=2)
        
        save_progress(folder_name, index + 1)
        index += 1
    
    if index >= len(folder_data):
        return None, f"✅ {folder_name} 标注完成！", gr.update(visible=False), str(index)
    
    next_img = Image.open(folder_data[index]["filename"])
    return next_img, f"{index+1}/{len(folder_data)}", gr.update(visible=True), str(index)

def change_folder(folder_name):
    if folder_name not in all_data:
        imgs = image_files[folder_name]
        all_data[folder_name] = [{
            "filename": f,
            "page_num": extract_page_number(os.path.basename(f)),
            "label_en": None
        } for f in imgs]
    
    folder_data = all_data[folder_name]
    last_index = min(int(load_progress(folder_name)), len(folder_data)-1)
    
    current_img = Image.open(folder_data[last_index]["filename"])
    return current_img, f"{last_index+1}/{len(folder_data)}", gr.update(visible=True), str(last_index)

with gr.Blocks(title="PPT图像标注工具") as demo:
    gr.Markdown("## PPT图像标注工具 (按页码顺序)")
    
    current_index = gr.State("0")
    
    with gr.Row():
        folder_select = gr.Dropdown(
            choices=natsorted(list(image_files.keys())),
            label="选择PPT文件夹",
            value=natsorted(list(image_files.keys()))[0] if image_files else None
        )
        
    with gr.Row():
        with gr.Column(scale=2):
            image_display = gr.Image(type="pil", label="当前PPT页面")
            progress = gr.Textbox(label="标注进度")
        with gr.Column(scale=1):
            radio = gr.Radio(
                LABELS,
                label="选择图像类型",
                value=LABELS[0]
            )
            submit_btn = gr.Button("提交并下一页", variant="primary")
            with gr.Accordion("高级选项", open=False):
                stats_btn = gr.Button("显示标签统计")
                resume_btn = gr.Button("跳转到上次标注位置")
            stats_display = gr.JSON(label="标签统计", visible=False)
            status_output = gr.Textbox(label="状态", visible=False)

    folder_select.change(
        change_folder,
        inputs=folder_select,
        outputs=[image_display, progress, radio, current_index]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[stats_display, status_output]
    )
    
    submit_btn.click(
        handle_labeling,
        inputs=[radio, folder_select, current_index],
        outputs=[image_display, progress, radio, current_index]
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=False)),
        outputs=[stats_display, status_output]
    )
    
    stats_btn.click(
        lambda: (dict(tag_counter), gr.update(visible=True), gr.update(visible=False)),
        outputs=[stats_display, stats_display, status_output]
    )
    
    resume_btn.click(
        lambda f: (gr.update(visible=False), f"已跳转到文件夹 {f} 的上次标注位置", gr.update(visible=True)),
        inputs=[folder_select],
        outputs=[stats_display, status_output, status_output]
    )

    if image_files:
        demo.load(
            lambda: change_folder(natsorted(list(image_files.keys()))[0]),
            outputs=[image_display, progress, radio, current_index]
        )

demo.launch(server_name="0.0.0.0", server_port=port)