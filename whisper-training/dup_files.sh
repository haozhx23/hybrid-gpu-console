#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "Usage: \$0 <source_file> <number_of_copies>"
    exit 1
fi

SOURCE_FILE=\$1
NUM_COPIES=\$2

# 检查源文件是否存在
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' does not exist"
    exit 1
fi

# 获取文件扩展名
EXTENSION="${SOURCE_FILE##*.}"

# 复制文件
for ((i=1; i<=$NUM_COPIES; i++))
do
    # 使用 UUID 生成唯一文件名
    NEW_NAME=$(uuidgen)."$EXTENSION"
    cp "$SOURCE_FILE" "$NEW_NAME"
    echo "Created copy $i: $NEW_NAME"
done

echo "Completed creating $NUM_COPIES copies"
