#!/bin/bash

METHOD=$1         # 第1个参数：method1
PHASE=$2          # 第3个参数：main 或 test        # 第4个参数：epoch 数
BATCH_SIZE=$3     # 第5个参数：batch size

PROJECT_DIR="$(dirname $(dirname "$0"))"
SRC_DIR="${PROJECT_DIR}/src"
MAIN_PY="${SRC_DIR}/${METHOD}/${PHASE}.py"

python "$MAIN_PY" \
  --batch_size "$BATCH_SIZE"
