#!/bin/bash
#!/bin/bash

# 获取当前脚本所在目录（script 文件夹）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 获取项目根目录（假设是 script 的上一级目录）
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# src 路径
SRC_DIR="$PROJECT_DIR/src"

# 解析前两个参数：方法名 METHOD 和阶段 PHASE
METHOD="$1"
PHASE="$2"

# 检查是否提供了 METHOD 和 PHASE
if [ -z "$METHOD" ] || [ -z "$PHASE" ]; then
    echo "❌ 错误：必须提供方法名和阶段。"
    echo "用法: $0 <METHOD> <PHASE> [OPTIONS]"
    echo "示例: $0 CTEN train --batch_size 8 --use_cuda"
    exit 1
fi

# 移除前两个参数，保留其余参数传给 Python
shift 2
EXTRA_ARGS="$@"

# 构建完整的 Python 主程序路径
MAIN_PY="${SRC_DIR}/${METHOD}/${PHASE}.py"

# 检查主程序是否存在
if [ ! -f "$MAIN_PY" ]; then
    echo "❌ 错误：找不到主程序 '$MAIN_PY'"
    exit 1
fi

# 输出运行信息
echo "🚀 正在运行: $MAIN_PY"
echo "📄 参数: $EXTRA_ARGS"

# 执行 Python 命令
python "$MAIN_PY" $EXTRA_ARGS
