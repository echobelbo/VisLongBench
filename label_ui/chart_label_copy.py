import os
import json
import gradio as gr
from collections import defaultdict

# ===== 配置 =====
data_root = "./data/slideshare/images"  # 存放 PPT 图片的目录
output_json = "./data/slideshare/query/chart_label.json"
categories = ["chart", "flowchart", "table", "normal"]

# ===== 数据操作 =====
def load_ppt_names():
    """扫描所有 PPT 文件夹名"""
    return sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

def load_annotations():
    """加载已有标注"""
    if not os.path.exists(output_json):
        return {}
    with open(output_json, "r", encoding="utf-8") as f:
        return json.load(f)

def save_annotations(data):
    """保存标注"""
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_annotation(ppt_name, start_page, end_page, category):
    """添加标注"""
    data = load_annotations()

    # 如果 PPT 不存在则初始化
    if ppt_name not in data:
        data[ppt_name] = []

    # 自动生成 group_id
    existing_ids = [item["group_id"] for item in data[ppt_name]]
    next_id = max(existing_ids) + 1 if existing_ids else 1

    new_item = {
        "group_id": next_id,
        "start_page": int(start_page),
        "end_page": int(end_page),
        "category": category
    }
    data[ppt_name].append(new_item)

    save_annotations(data)
    return f"✅ 已添加: {ppt_name} ({start_page}-{end_page}, {category})", update_stats(ppt_name, data)

def update_stats(ppt_name, data=None):
    """统计当前 PPT 标注情况"""
    if data is None:
        data = load_annotations()
    if ppt_name not in data:
        return f"当前标注统计（{ppt_name}）：\n暂无数据"

    stats = defaultdict(int)
    total = len(data[ppt_name])
    for item in data[ppt_name]:
        stats[item["category"]] += 1

    stats_text = f"当前标注统计（{ppt_name}）:\n总计: {total} 组\n"
    for cat in categories:
        stats_text += f"- {cat}: {stats[cat]}\n"
    return stats_text

def reload_data(ppt_name):
    """重新加载数据"""
    return update_stats(ppt_name)

# ===== Gradio UI =====
with gr.Blocks() as demo:
    gr.Markdown("## 📑 PPT 重点图像标注工具（JSON 存储版）")

    with gr.Row():
        ppt_name = gr.Dropdown(choices=load_ppt_names(), label="选择 PPT 名")
        start_page = gr.Number(label="起始页", precision=0)
        end_page = gr.Number(label="结束页", precision=0)
        category = gr.Dropdown(choices=categories, label="分类")

    with gr.Row():
        add_btn = gr.Button("添加标注", variant="primary")
        reload_btn = gr.Button("重新加载标注数据")
        stats_box = gr.Textbox(label="统计信息", interactive=False)

    log_box = gr.Textbox(label="操作日志", interactive=False)

    # 事件绑定
    add_btn.click(
        add_annotation,
        inputs=[ppt_name, start_page, end_page, category],
        outputs=[log_box, stats_box]
    )

    ppt_name.change(
        reload_data,
        inputs=[ppt_name],
        outputs=[stats_box]
    )

    reload_btn.click(
        reload_data,
        inputs=[ppt_name],
        outputs=[stats_box]
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
