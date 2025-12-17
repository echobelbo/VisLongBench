import gradio as gr
import json
import os
from collections import defaultdict

output_path = "./data/tutorial/chunk.json"
paragraph_data = defaultdict(list)

def add_paragraph(pdf_name, start, end, title):
    if not pdf_name.strip():
        return "❌ 文件名不能为空"
    if start > end:
        return "❌ 开始页不能大于结束页"
    paragraph_data[pdf_name].append({
        "start": int(start),
        "end": int(end),
        "title": title.strip()
    })
    return f"✅ 添加段落：[{pdf_name}] {start}-{end} - {title}"

def export_json():
    with open(output_path, "w") as f:
        json.dump(paragraph_data, f, indent=2)
    return f"✅ JSON 已保存到 {output_path}"

def import_json():
    if not os.path.exists(output_path):
        return "⚠️ 未找到已有文件", ""
    with open(output_path, "r") as f:
        data = json.load(f)
    paragraph_data.clear()
    for k, v in data.items():
        paragraph_data[k] = v
    return f"✅ 成功加载 {output_path}", json.dumps(paragraph_data, indent=2)

def show_current():
    return json.dumps(paragraph_data, indent=2)

def clear_all():
    paragraph_data.clear()
    return "✅ 已清空所有数据"

with gr.Blocks(title="段落结构标注工具") as demo:
    gr.Markdown("### 📘 幻灯片段落结构生成器")

    with gr.Row():
        pdf_name = gr.Textbox(label="PDF 文件名")
        start_page = gr.Number(label="开始页码", precision=0)
        end_page = gr.Number(label="结束页码", precision=0)
        title = gr.Textbox(label="段落标题")

    add_button = gr.Button("➕ 添加段落")
    status = gr.Textbox(label="状态", interactive=False)

    with gr.Row():
        export_button = gr.Button("💾 保存 JSON")
        import_button = gr.Button("📂 加载 JSON")
        clear_button = gr.Button("🧹 清空数据")

    show_button = gr.Button("📋 查看当前数据")
    json_view = gr.Code(label="当前结构 JSON", language="json")

    add_button.click(
        add_paragraph,
        inputs=[pdf_name, start_page, end_page, title],
        outputs=status
    )
    export_button.click(export_json, outputs=status)
    import_button.click(import_json, outputs=[status, json_view])
    clear_button.click(clear_all, outputs=status)
    show_button.click(show_current, outputs=json_view)

demo.launch(server_name="0.0.0.0", server_port=7861)
 