#!/usr/bin/env python3
"""从 results/chunks.csv 构建图网络数据，并预计算布局。

输出 web/graph_data.json，供 web/index.html 直接读取。

用法：
    .venv/bin/python build_graph.py --max-rows 10
    .venv/bin/python build_graph.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - 运行时提示
    print("缺少 networkx，请先激活 .venv，并安装 requirements.txt", file=sys.stderr)
    raise SystemExit(2) from exc


DEFAULT_INPUT = "results/chunks.csv"
DEFAULT_OUTPUT = "web/graph_data.json"


TYPE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("人物", ("人物", "总统", "主席", "总理", "部长", "官员", "经济学家", "学者", "作者", "教授",
              "议员", "财长", "国务卿", "银行家", "CEO", "创始人", "将军")),
    ("组织机构", ("组织", "机构", "企业", "公司", "银行", "政府", "央行", "美联储",
                  "委员会", "部门", "基金", "交易所", "联盟", "协会", "议会", "国会", "联储")),
    ("国家地区", ("国家", "地区", "地点", "城市", "州", "领土", "中国", "美国", "俄罗斯",
                  "欧洲", "亚洲", "日本", "德国", "法国", "英国", "全球")),
    ("金融工具资产", ("债券", "国债", "票据", "证券", "股票", "基金", "期货", "期权",
                     "衍生品", "货币", "资产", "金融工具", "金融产品", "外汇", "黄金",
                     "存款", "贷款", "债务")),
    ("经济事件危机", ("事件", "危机", "违约", "战争", "恐慌", "衰退", "泡沫", "冲突")),
    ("政策制度", ("政策", "制度", "法案", "协议", "条例", "规则", "法律", "体系",
                  "计划", "条款", "央策")),
    ("金融经济概念", ("金融", "经济", "信用", "财政", "通胀", "通缩", "风险", "资本",
                     "投资", "交易", "融资", "流动性", "利率", "汇率", "收益率", "市场")),
    ("时间阶段", ("时间", "年代", "年份", "时期", "阶段", "世纪", "日期", "季度")),
]


def normalize_key(value: Any) -> str:
    return str(value or "").strip().casefold()


# 受控词表：把易混别名归并到规范名。仅对表内 key 生效，不影响其他实体归一化。
# 作用：让"一战"/"wwi"等变体并入"第一次世界大战"的根，避免 LLM 把关系端点写成错别名时
# 形成孤立节点或错连。id 含 # 后缀的生成规则（UnionFind/root_to_node_id）保持不变。
CANONICAL_ALIASES: dict[str, str] = {
    "一战": "第一次世界大战",
    "第一次世界大战": "第一次世界大战",
    "wwi": "第一次世界大战",
    "二战": "第二次世界大战",
    "第二次世界大战": "第二次世界大战",
    "wwii": "第二次世界大战",
    # 可继续按业务补充：如 "大萧条": "1929年大萧条" 等
}


def canonical_key(value: Any) -> str:
    """在 normalize_key 基础上，对受控词表内的别名做规范化；表外原样归一。"""
    key = normalize_key(value)
    return normalize_key(CANONICAL_ALIASES.get(key, value))


def simplify_type(raw_type: str) -> str:
    """把 1975 个细碎 type 归并到少数量大类，方便着色和筛选。"""
    text = str(raw_type or "").strip()
    if not text or text == "未知":
        return "未知"
    for category, keywords in TYPE_RULES:
        for keyword in keywords:
            if keyword in text:
                return category
    return "其他"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        if key not in self.parent:
            self.parent[key] = key

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            next_key = self.parent[key]
            self.parent[key] = root
            key = next_key
        return root

    def union(self, key_a: str, key_b: str) -> None:
        root_a = self.find(key_a)
        root_b = self.find(key_b)
        if root_a == root_b:
            return
        # 字典序较短的优先成为 root，减少长尾节点 id
        if (len(root_a), root_a) <= (len(root_b), root_b):
            self.parent[root_b] = root_a
        else:
            self.parent[root_a] = root_b

    def normalize_roots(self) -> None:
        for key in list(self.parent):
            self.find(key)


def safe_json(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=0, help="只处理前 N 行；0 表示全量")
    parser.add_argument("--max-evidence", type=int, default=8, help="每个节点最多保留多少条 evidence")
    parser.add_argument("--seed", type=int, default=42, help="布局随机种子")
    return parser.parse_args()


def read_chunks(input_path: Path, max_rows: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_no, row in enumerate(reader, start=1):
            if max_rows and line_no > max_rows:
                break
            parsed = safe_json(row.get("summary_json"))
            if parsed is None:
                continue
            chunk_id = str(row.get("id") or f"row-{line_no}")
            for entity in parsed.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                records.append(
                    {
                        "kind": "entity",
                        "chunk_id": chunk_id,
                        "name": entity.get("name"),
                        "type": entity.get("type") or "未知",
                        "aliases": entity.get("aliases") or [],
                        "evidence": entity.get("evidence") or "",
                    }
                )
            for relation in parsed.get("relationships", []):
                if not isinstance(relation, dict):
                    continue
                records.append(
                    {
                        "kind": "relation",
                        "chunk_id": chunk_id,
                        "source": relation.get("source"),
                        "target": relation.get("target"),
                        "relation": relation.get("relation") or "未知",
                        "evidence": relation.get("evidence") or "",
                        # #2 逐字校验用：evidence 必须是本切片原文（content）的子串
                        "content": row.get("content") or "",
                    }
                )
    return records


def build_nodes_and_edges(records: list[dict[str, Any]], max_evidence: int):
    entity_records = [record for record in records if record["kind"] == "entity"]
    relation_records = [record for record in records if record["kind"] == "relation"]

    uf = UnionFind()
    entity_keys: list[list[str]] = []
    for record in entity_records:
        # #4 用 canonical_key 代替 normalize_key，使受控词表内的别名（一战/二战…）并入规范 root
        name_key = canonical_key(record["name"])
        alias_keys = [
            canonical_key(alias)
            for alias in record["aliases"]
            if canonical_key(alias)
        ]
        keys = list(dict.fromkeys([name_key, *alias_keys]))
        entity_keys.append(keys)
        for key in keys:
            uf.add(key)
        for other in keys[1:]:
            uf.union(keys[0], other)

    uf.normalize_roots()

    root_display_counter: dict[str, Counter[str]] = defaultdict(Counter)
    root_display_names: dict[str, set[str]] = defaultdict(set)
    root_type_counter: dict[str, Counter[str]] = defaultdict(Counter)
    root_evidence: dict[str, list[str]] = defaultdict(list)
    root_evidence_seen: dict[str, set[str]] = defaultdict(set)
    root_mentions: Counter[str] = Counter()
    root_chunk_ids: dict[str, set[str]] = defaultdict(set)

    for record, keys in zip(entity_records, entity_keys):
        root = uf.find(keys[0])
        display_name = str(record["name"] or "").strip()
        if not display_name:
            continue
        root_display_counter[root][display_name] += 1
        root_display_names[root].add(display_name)
        for alias in record["aliases"]:
            alias_display = str(alias).strip()
            if alias_display:
                root_display_names[root].add(alias_display)
        raw_type = str(record["type"] or "未知").strip()
        root_type_counter[root][raw_type] += 1
        evidence = str(record.get("evidence") or "").strip()
        if evidence and evidence not in root_evidence_seen[root]:
            root_evidence_seen[root].add(evidence)
            if len(root_evidence[root]) < max_evidence:
                root_evidence[root].append(evidence)
        root_mentions[root] += 1
        root_chunk_ids[root].add(record["chunk_id"])

    root_by_key: dict[str, str] = {}
    for key, root in uf.parent.items():
        root_by_key[key] = uf.find(root)

    used_node_ids: set[str] = set()
    node_by_root: dict[str, dict[str, Any]] = {}
    root_to_node_id: dict[str, str] = {}

    for root, display_counter in root_display_counter.items():
        canonical_display = display_counter.most_common(1)[0][0]
        node_id = canonical_display
        suffix = 1
        while node_id in used_node_ids:
            suffix += 1
            node_id = f"{canonical_display}#{suffix}"
        used_node_ids.add(node_id)
        root_to_node_id[root] = node_id

        aliases = sorted(
            (name for name in root_display_names[root] if name != canonical_display),
            key=lambda name: -root_display_counter[root][name],
        )
        raw_type = root_type_counter[root].most_common(1)[0][0]
        node_by_root[root] = {
            "id": node_id,
            "name": canonical_display,
            "type": raw_type,
            "type_simple": simplify_type(raw_type),
            "aliases": aliases[:20],
            "evidence": root_evidence[root][:max_evidence],
            "mentions": root_mentions[root],
            "degree": 0,
            "in_degree": 0,
            "out_degree": 0,
            "weighted_degree": 0,
            "size": 6,
            "x": 0.0,
            "y": 0.0,
            "seed_weight": root_mentions[root],
        }

    def resolve_node(value: Any, node_id: str | None = None) -> str:
        raw_text = str(value or "").strip()
        # #4 端点同样走 canonical_key，与实体 root 的归一规则保持一致
        key = canonical_key(raw_text)
        root = root_by_key.get(key)
        if root in root_to_node_id:
            return root_to_node_id[root]
        if raw_text:
            candidate = raw_text
        else:
            candidate = node_id or f"orphan-{len(root_to_node_id)}"
        node_id_value = candidate
        suffix = 1
        while node_id_value in used_node_ids:
            suffix += 1
            node_id_value = f"{candidate}#{suffix}"
        used_node_ids.add(node_id_value)
        root_to_node_id[key] = node_id_value
        node_by_root[key] = {
            "id": node_id_value,
            "name": candidate,
            "type": "未知",
            "type_simple": "未知",
            "aliases": [],
            "evidence": [],
            "mentions": 0,
            "degree": 0,
            "in_degree": 0,
            "out_degree": 0,
            "weighted_degree": 0,
            "size": 6,
            "x": 0.0,
            "y": 0.0,
            "seed_weight": 0,
        }
        return node_id_value

    pair_relations: dict[
        tuple[str, str], dict[str, Any]
    ] = defaultdict(
        lambda: {
            "source": "",
            "target": "",
            "relations": {},
            "pair_weight": 0,
        }
    )

    dropped_relations = 0
    for record in relation_records:
        # #1 端点必须命中已建实体（root_by_key）；否则丢弃该边，不调用 resolve_node 造孤儿节点。
        # 前提：此兜底依赖 #3（summarize_pipeline 提示词强制 target∈entities）才安全——
        # 若模型漏抽实体会导致正确边也被丢弃，故必须与提示词强化配套使用。
        src_key = canonical_key(record.get("source"))
        tgt_key = canonical_key(record.get("target"))
        if src_key not in root_by_key or tgt_key not in root_by_key:
            dropped_relations += 1
            continue
        source = resolve_node(record.get("source"), "source")
        target = resolve_node(record.get("target"), "target")
        pair_key = (source, target)
        pair = pair_relations[pair_key]
        pair["source"] = source
        pair["target"] = target

        relation_name = str(record.get("relation") or "未知").strip() or "未知"
        relation_key = relation_name.casefold()
        relation_entry = pair["relations"].setdefault(
            relation_key,
            {
                "relation": relation_name,
                "evidence": [],
                "count": 0,
            },
        )
        relation_entry["count"] += 1
        evidence = str(record.get("evidence") or "").strip()
        # #2 逐字校验：evidence 必须是对应切片原文（content）的子串，否则丢弃该证据、保留关系本身
        content = str(record.get("content") or "")
        if evidence and content and evidence not in content:
            evidence = ""
        if evidence and evidence not in relation_entry["evidence"]:
            if len(relation_entry["evidence"]) < max_evidence:
                relation_entry["evidence"].append(evidence)
        pair["pair_weight"] += 1

    nodes = list(node_by_root.values())

    graph = nx.DiGraph()
    node_ids = [node["id"] for node in nodes]
    graph.add_nodes_from(node_ids)

    edges: list[dict[str, Any]] = []
    for (source, target), pair in pair_relations.items():
        weight = int(pair["pair_weight"])
        graph.add_edge(source, target, weight=weight)
        relations = sorted(
            pair["relations"].values(),
            key=lambda item: -item["count"],
        )
        edges.append(
            {
                "id": f"{source}__{target}",
                "source": source,
                "target": target,
                "weight": weight,
                "relations": relations,
            }
        )

    undirected = graph.to_undirected()
    degree_map = dict(undirected.degree(weight=None))
    weighted_degree_map = dict(undirected.degree(weight="weight"))
    in_degree_map = dict(graph.in_degree(weight=None))
    out_degree_map = dict(graph.out_degree(weight=None))

    for node in nodes:
        node_id = node["id"]
        node["degree"] = int(degree_map.get(node_id, 0))
        node["weighted_degree"] = float(weighted_degree_map.get(node_id, 0.0))
        node["in_degree"] = int(in_degree_map.get(node_id, 0))
        node["out_degree"] = int(out_degree_map.get(node_id, 0))

    if nodes:
        max_weighted_degree = max((node["weighted_degree"] for node in nodes), default=1.0) or 1.0
        for node in nodes:
            ratio = node["weighted_degree"] / max_weighted_degree
            node["size"] = round(6.0 + 30.0 * math.sqrt(ratio), 2)

    return nodes, edges, graph.to_undirected(), {"dropped_relations": dropped_relations}


def compute_community_layout(graph: nx.Graph, seed: int) -> dict[Any, tuple[float, float]]:
    """用 Louvain 分簇 + 簇内 spring layout，避免全局弹簧对 3500 节点过慢。"""
    if graph.number_of_nodes() == 0:
        return {}

    try:
        communities = sorted(
            nx.community.louvain_communities(graph, seed=seed, weight="weight"),
            key=len,
            reverse=True,
        )
    except Exception:
        communities = [set(graph.nodes())]

    community_nodes: list[set[Any]] = [set(nodes) for nodes in communities]
    node_to_community: dict[Any, int] = {}
    for community_index, nodes in enumerate(community_nodes):
        for node in nodes:
            node_to_community[node] = community_index

    meta_graph = nx.Graph()
    meta_graph.add_nodes_from(range(len(community_nodes)))
    for u, v, data in graph.edges(data=True):
        community_u = node_to_community.get(u)
        community_v = node_to_community.get(v)
        if community_u is None or community_v is None or community_u == community_v:
            continue
        weight = float(data.get("weight", 1.0))
        if meta_graph.has_edge(community_u, community_v):
            meta_graph[community_u][community_v]["weight"] += weight
        else:
            meta_graph.add_edge(community_u, community_v, weight=weight)

    if len(community_nodes) > 1:
        meta_scale = max(1500.0, 180.0 * len(community_nodes))
        meta_positions = nx.spring_layout(
            meta_graph,
            seed=seed,
            iterations=100,
            scale=meta_scale,
        )
    else:
        meta_positions = {0: (0.0, 0.0)}

    final_positions: dict[Any, tuple[float, float]] = {}
    for community_index, nodes in enumerate(community_nodes):
        subgraph = graph.subgraph(nodes)
        center_x, center_y = meta_positions[community_index]
        radius = max(50.0, 6.0 * math.sqrt(len(nodes)))

        if len(nodes) == 1:
            local_positions = {next(iter(nodes)): (0.0, 0.0)}
        elif len(nodes) <= 350:
            local_positions = nx.spring_layout(
                subgraph,
                seed=seed,
                iterations=35,
                scale=radius,
                center=(0.0, 0.0),
            )
        elif len(nodes) <= 1000:
            local_positions = nx.spring_layout(
                subgraph,
                seed=seed,
                iterations=15,
                scale=radius,
                center=(0.0, 0.0),
            )
        else:
            # 超大簇不再跑重弹簧，改为 degree 排序的圆环，防止布局时间爆炸
            ordered = sorted(subgraph.nodes(), key=lambda n: subgraph.degree(n), reverse=True)
            radius = max(80.0, 8.0 * math.sqrt(len(ordered)))
            local_positions = {
                node: (
                    radius * math.cos(2 * math.pi * index / len(ordered)),
                    radius * math.sin(2 * math.pi * index / len(ordered)),
                )
                for index, node in enumerate(ordered)
            }

        for node, (local_x, local_y) in local_positions.items():
            final_positions[node] = (center_x + local_x, center_y + local_y)

    return final_positions


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"找不到输入文件：{input_path}", file=sys.stderr)
        return 2

    if args.max_rows:
        print(f"读取前 {args.max_rows} 行：{input_path}")
    else:
        print(f"读取全量数据：{input_path}")

    records = read_chunks(input_path, args.max_rows)
    print(f"读取到 {len(records)} 条实体/关系记录")

    nodes, edges, undirected, stats = build_nodes_and_edges(records, args.max_evidence)
    print(f"节点数（去重后）：{len(nodes)}")
    print(f"关系对数（去重后）：{len(edges)}")
    # 端点未命中实体的被丢弃关系数；持续偏大说明模型漏抽实体或提示词未强制 target∈entities
    print(f"被丢弃的孤立端点关系数（端点未命中实体）：{stats.get('dropped_relations', 0)}")
    print("开始预计算布局（Louvain 分簇 + 簇内 spring layout）...")

    positions = compute_community_layout(undirected, args.seed)
    for node in nodes:
        node_id = node["id"]
        node["x"], node["y"] = positions.get(node_id, (0.0, 0.0))

    nodes.sort(key=lambda item: (-item["degree"], -item["mentions"], item["id"]))

    type_counter: Counter[str] = Counter(node["type_simple"] for node in nodes)
    relation_counter: Counter[str] = Counter()
    for edge in edges:
        for relation in edge["relations"]:
            relation_counter[relation["relation"]] += relation["count"]

    type_colors = {
        "人物": "#f28e2b",
        "组织机构": "#e15759",
        "国家地区": "#76b7b2",
        "金融工具资产": "#4e79a7",
        "经济事件危机": "#edc948",
        "政策制度": "#59a14f",
        "金融经济概念": "#b07aa1",
        "时间阶段": "#ff9da7",
        "未知": "#9aa0a6",
        "其他": "#79706e",
    }

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(input_path),
            "max_rows": args.max_rows or None,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "type_colors": {key: type_colors.get(key, "#888888") for key in type_counter},
            "type_distribution": dict(type_counter.most_common()),
            "relation_distribution": dict(relation_counter.most_common()),
            "layout": "louvain-communities-spring",
        },
        "nodes": nodes,
        "edges": edges,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"已写入：{output_path}")
    print(f"实体类型分布：{dict(type_counter.most_common(12))}")
    print(f"关系类型分布：{dict(relation_counter.most_common(12))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
