#!/bin/bash
# 用法: ./find_record.sh <id> <csv文件>
# 示例: ./find_record.sh 7c5d9121-2024-5d72-a670-3d3dd6dc4785 chunks.csv

set -e

if [ $# -ne 2 ]; then
    echo "用法: $0 <id> <csv文件>" >&2
    exit 1
fi

TARGET_ID="$1"
CSV_FILE="$2"

if [ ! -f "$CSV_FILE" ]; then
    echo "文件不存在: $CSV_FILE" >&2
    exit 1
fi

awk -v id="$TARGET_ID" '
BEGIN { found = 0 }
/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12},/ {
    if (found) exit
    if ($0 ~ "^" id ",") found = 1
}
found { print }
END {
    if (!found) {
        print "未找到 id: " id > "/dev/stderr"
        exit 2
    }
}
' "$CSV_FILE"
