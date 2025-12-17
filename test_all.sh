#!/bin/bash

# ---------------------------------------------
# 自动遍历所有 query_type 与 model 组合运行：
#   python generate.py --query_type="xxx" --model="yyy"
# ---------------------------------------------

# 可选 query_type
QUERY_TYPES=("detail" "direct" "chapter")

# 可选 model
MODELS=(
    "qwen2.5-vl-7b-instruct"
    "gemini-2.5-pro"
    "claude-3-5-sonnet-latest"
    # "Qwen/Qwen2.5-VL-72B-Instruct"
)

# 输出日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "🚀 开始遍历所有组合..."

for qt in "${QUERY_TYPES[@]}"; do
    for model in "${MODELS[@]}"; do
        
        echo "---------------------------------------------"
        echo "▶ Running query_type = $qt , model = $model"
        echo "---------------------------------------------"

        LOG_FILE="${LOG_DIR}/${qt}__$(echo $model | sed 's|/|_|g').log"

        # 执行命令并记录日志
        python generate.py --query_type="$qt" --model="$model" | tee "$LOG_FILE"

        echo "✔ 已完成：query_type=$qt, model=$model"
        echo
    done
done

echo "🎉 全部运行完成！日志保存在 logs/ 目录。"
