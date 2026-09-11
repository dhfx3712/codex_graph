#!/usr/bin/env python3
"""统计 results/chunks.csv 中实体与关联关系数量，并进行节点量估算。

用法：
    # 先用 10 行数据验证脚本
    python3 count_graph_stats.py --limit 10

    # 全量统计
    python3 count_graph_stats.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CSV = "results/chunks.csv"


def normalize(value: Any, case_sensitive: bool) -> str:
    """统一名称比较时的空值处理和可选大小写处理。"""
    text = str(value or "").strip()
    if not case_sensitive:
        text = text.casefold()
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 行数据；0 表示全量处理",
    )
    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="名称去重时区分大小写；默认不区分",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="额外输出一份 JSON 结果，方便后续程序使用",
    )
    parser.add_argument(
        "--numbers-only",
        action="store_true",
        help="只输出各项统计个数，不输出 Top20 和类型分布明细",
    )
    return parser.parse_args()


def safe_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def iter_rows(csv_path: Path, limit: int):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if limit and index > limit:
                return
            yield index, row


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"找不到文件：{csv_path}", file=sys.stderr)
        return 2

    rows = 0
    parsed_json_rows = 0
    missing_json_rows = 0
    bad_json_rows = 0

    entity_records = 0
    relation_records = 0

    entity_name_counter: Counter[str] = Counter()
    entity_key_counter: Counter[tuple[str, str]] = Counter()
    entity_type_counter: Counter[str] = Counter()
    relation_key_counter: Counter[tuple[str, str, str]] = Counter()
    relation_type_counter: Counter[str] = Counter()

    entity_name_set: set[str] = set()
    relation_endpoint_set: set[str] = set()

    malformed_examples: list[tuple[int, str]] = []

    case_sensitive = args.case_sensitive

    for line_no, row in iter_rows(csv_path, args.limit):
        rows += 1
        raw = row.get("summary_json")
        parsed = safe_json(raw)

        if not raw:
            missing_json_rows += 1
            continue

        if parsed is None:
            bad_json_rows += 1
            if len(malformed_examples) < 5:
                malformed_examples.append((line_no, raw[:120]))
            continue

        parsed_json_rows += 1

        entities = parsed.get("entities", [])
        relations = parsed.get("relationships", [])

        if not isinstance(entities, list):
            entities = []
        if not isinstance(relations, list):
            relations = []

        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = normalize(entity.get("name"), case_sensitive)
            if not name:
                continue

            entity_type = normalize(entity.get("type") or "未分类", case_sensitive)
            entity_records += 1
            entity_name_counter[name] += 1
            entity_key_counter[(name, entity_type)] += 1
            entity_type_counter[entity_type] += 1
            entity_name_set.add(name)

        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = normalize(relation.get("source"), case_sensitive)
            target = normalize(relation.get("target"), case_sensitive)
            relation_type = normalize(relation.get("relation") or "未知关系", case_sensitive)
            if not source or not target:
                continue

            relation_records += 1
            relation_key_counter[(source, target, relation_type)] += 1
            relation_type_counter[relation_type] += 1
            relation_endpoint_set.add(source)
            relation_endpoint_set.add(target)

    unknown_endpoint_count = len(relation_endpoint_set - entity_name_set)
    estimated_node_count = len(entity_name_set | relation_endpoint_set)

    result = {
        "input": {
            "file": str(csv_path),
            "limit": args.limit or None,
            "rows_read": rows,
            "rows_with_valid_json": parsed_json_rows,
            "rows_without_summary_json": missing_json_rows,
            "rows_with_bad_json": bad_json_rows,
        },
        "entities": {
            "records_total": entity_records,
            "dedup_by_name": len(entity_name_counter),
            "dedup_by_name_and_type": len(entity_key_counter),
            "type_count": len(entity_type_counter),
        },
        "relationships": {
            "records_total": relation_records,
            "dedup_by_source_target_relation": len(relation_key_counter),
            "relation_type_count": len(relation_type_counter),
        },
        "node_estimate": {
            "recommended_nodes_by_name": estimated_node_count,
            "entity_names_only": len(entity_name_set),
            "endpoint_names_not_in_entities": unknown_endpoint_count,
        },
    }

    print("=" * 64)
    print("实体 / 关系统计")
    print("=" * 64)
    print(f"文件：{csv_path}")
    print(f"读取行数：{rows}")
    if args.limit:
        print(f"模式：仅前 {args.limit} 行")
    else:
        print("模式：全量")
    print()
    print(f"summary_json 有效行数：{parsed_json_rows}")
    print(f"summary_json 缺失行数：{missing_json_rows}")
    print(f"summary_json 解析失败行数：{bad_json_rows}")
    if malformed_examples:
        print("解析失败示例：")
        for line_no, snippet in malformed_examples:
            print(f"  - 第 {line_no} 行：{snippet}")
    print()

    print("【实体】")
    print(f"  实体记录总数（去重前）：{entity_records}")
    print(f"  按 name 去重：{result['entities']['dedup_by_name']}")
    print(f"  按 name+type 去重：{result['entities']['dedup_by_name_and_type']}")
    print(f"  实体 type 数：{result['entities']['type_count']}")
    print()

    print("【关联关系】")
    print(f"  关系记录总数（去重前）：{relation_records}")
    print(
        "  按 source+target+relation 去重："
        f"{result['relationships']['dedup_by_source_target_relation']}"
    )
    print(f"  关系类型数：{result['relationships']['relation_type_count']}")
    print()

    print("【节点量估算】")
    print(
        "  建议节点数（entity name ∪ relationship 端点）："
        f"{estimated_node_count}"
    )
    print(f"  其中仅由实体产生的 name 数：{len(entity_name_set)}")
    print(f"  出现在关系中但不在实体列表的端点数：{unknown_endpoint_count}")
    print()

    if not args.numbers_only and entity_name_counter:
        print("【重复次数最高的实体 name】（前 20）")
        for name, count in entity_name_counter.most_common(20):
            print(f"  {count:>6} 次  {name}")
        print()

    if not args.numbers_only and relation_key_counter:
        print("【重复次数最高的关系】（前 20）")
        for (source, target, relation_type), count in relation_key_counter.most_common(20):
            print(f"  {count:>6} 次  {source} -> {target} [{relation_type}]")
        print()

    if not args.numbers_only and entity_type_counter:
        print("【实体类型分布】（前 20）")
        for entity_type, count in entity_type_counter.most_common(20):
            print(f"  {count:>6} 个实体记录  {entity_type}")
        print()

    if not args.numbers_only and relation_type_counter:
        print("【关系类型分布】（前 20）")
        for relation_type, count in relation_type_counter.most_common(20):
            print(f"  {count:>6} 条关系记录  {relation_type}")
        print()

    if args.json:
        enhanced = dict(result)
        if not args.numbers_only:
            enhanced.update(
                {
                    "entity_type_distribution": dict(entity_type_counter.most_common()),
                    "relation_type_distribution": dict(relation_type_counter.most_common()),
                    "top_entities": [
                        {"name": name, "count": count}
                        for name, count in entity_name_counter.most_common(20)
                    ],
                    "top_relations": [
                        {
                            "source": source,
                            "target": target,
                            "relation": relation_type,
                            "count": count,
                        }
                        for (source, target, relation_type), count in relation_key_counter.most_common(20)
                    ],
                }
            )
        print("【JSON 结果】")
        print(json.dumps(enhanced, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
