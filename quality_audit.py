#!/usr/bin/env python3
"""chunks.csv 质量审计：零 LLM 成本，量化「抽取质量」与「源文本质量」。

用法：
  python quality_audit.py results/chunks.csv [--max-rows N]

输出：
  - 控制台：整体汇总 + 按文章的质量分（低分文章即重点整改对象）
  - /tmp/quality_by_article.csv：每篇文章的分项指标

为什么需要它（回答"要不要预估源文本质量"）：
  冒烟测试里 97.7% 的 evidence 被逐字校验拦下、12.3% 的关系端点未命中实体，
  这两个数字本身就是「抽取质量」的量化信号，而非「源文本质量」。但二者常被混为一谈。
  本脚本把两者拆开，并额外给出源文本启发式（正文长度、junk 切片率），让你在花
  LLM token 全量重抽之前，先用一次只读扫描评估：
    (a) 问题主要是抽取没遵守提示词（→ 重跑 summarize_pipeline 即可），还是
    (b) 源文本本身太短/太碎/无实体（→ 该先修源文本或剔除 junk 切片）。

指标定义：
  - evidence 逐字合规率：实体/关系的 evidence 在压缩空白、casefold 后是 content 子串的比例
  - 关系端点覆盖度：关系的 source/target 归一化后落在实体 name/aliases 集合的比例
  - junk 切片率：无实体或正文 < 40 字 的切片占比（源文本/抽取双差信号）
  - 质量分 = 端点覆盖*0.5 + evidence合规*0.3 + (1-junk)*0.2   （0~1，越高越好）
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict

csv.field_size_limit(sys.maxsize)


def norm(value: str) -> str:
    """压缩所有空白并 casefold——与 build_graph.canonical_key 的归一口径一致。"""
    return re.sub(r"\s+", "", str(value or "")).casefold()


def chunk_metrics(row: dict) -> dict:
    content = row.get("content") or ""
    try:
        parsed = json.loads(row.get("summary_json") or "")
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if not isinstance(parsed, dict):
        return {"n_ent": 0, "n_rel": 0, "ev_total": 0, "ev_ok": 0,
                "ep_total": 0, "ep_ok": 0, "len": len(content)}

    entities = [e for e in parsed.get("entities", []) if isinstance(e, dict)]
    rels = [r for r in parsed.get("relationships", []) if isinstance(r, dict)]

    ent_keys = set()
    for e in entities:
        ent_keys.add(norm(e.get("name")))
        for a in (e.get("aliases") or []):
            ent_keys.add(norm(a))

    ev_total = ev_ok = 0
    for e in entities:
        ev = e.get("evidence") or ""
        if ev:
            ev_total += 1
            if norm(ev) and norm(ev) in norm(content):
                ev_ok += 1
    for r in rels:
        ev = r.get("evidence") or ""
        if ev:
            ev_total += 1
            if norm(ev) and norm(ev) in norm(content):
                ev_ok += 1

    ep_total = ep_ok = 0
    for r in rels:
        ep_total += 1
        s = norm(r.get("source"))
        t = norm(r.get("target"))
        if s in ent_keys and t in ent_keys:
            ep_ok += 1

    return {"n_ent": len(entities), "n_rel": len(rels), "ev_total": ev_total,
            "ev_ok": ev_ok, "ep_total": ep_total, "ep_ok": ep_ok, "len": len(content)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="chunks.csv 路径")
    ap.add_argument("--max-rows", type=int, default=0, help="只审计前 N 行；0=全量")
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.input, encoding="utf-8-sig", newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle), start=1):
            if args.max_rows and i > args.max_rows:
                break
            rows.append(row)

    by_article: dict[str, list[dict]] = defaultdict(list)
    tot = Counter()
    for row in rows:
        m = chunk_metrics(row)
        by_article[row.get("article_id") or "unknown"].append(m)
        tot["n"] += 1
        tot["n_ent"] += m["n_ent"]
        tot["n_rel"] += m["n_rel"]
        tot["ev_total"] += m["ev_total"]
        tot["ev_ok"] += m["ev_ok"]
        tot["ep_total"] += m["ep_total"]
        tot["ep_ok"] += m["ep_ok"]
        tot["len"] += m["len"]
        if m["n_ent"] == 0 or m["len"] < 40:
            tot["junk"] += 1

    n = tot["n"] or 1
    ev_rate = tot["ev_ok"] / tot["ev_total"] if tot["ev_total"] else 1.0
    ep_rate = tot["ep_ok"] / tot["ep_total"] if tot["ep_total"] else 1.0
    junk_rate = tot["junk"] / n
    avg_ent = tot["n_ent"] / n
    avg_len = tot["len"] / n

    print(f"切片总数: {tot['n']}")
    print(f"实体/关系: {tot['n_ent']} / {tot['n_rel']}  (平均实体/切片 {avg_ent:.2f})")
    print(f"evidence 逐字合规率: {ev_rate*100:.1f}%  ({tot['ev_ok']}/{tot['ev_total']})")
    print(f"关系端点覆盖度(端点∈实体): {ep_rate*100:.1f}%  ({tot['ep_ok']}/{tot['ep_total']})")
    print(f"junk 切片率(无实体或正文<40字): {junk_rate*100:.1f}%")
    print(f"平均正文长度: {avg_len:.0f} 字")
    print()
    print("=== 按文章质量分（低分=重点整改；score=端点覆盖*0.5+ev合规*0.3+(1-junk)*0.2）===")

    out_rows = []
    for aid, ms in by_article.items():
        cn = len(ms)
        ev = sum(x["ev_ok"] for x in ms)
        evt = sum(x["ev_total"] for x in ms)
        ep = sum(x["ep_ok"] for x in ms)
        ept = sum(x["ep_total"] for x in ms)
        jk = sum(1 for x in ms if x["n_ent"] == 0 or x["len"] < 40)
        er = ev / evt if evt else 1.0
        epr = ep / ept if ept else 1.0
        jr = jk / cn
        score = epr * 0.5 + er * 0.3 + (1 - jr) * 0.2
        out_rows.append((aid, cn, round(epr * 100, 1), round(er * 100, 1),
                         round(jr * 100, 1), round(score, 3)))

    out_rows.sort(key=lambda r: r[5])
    header = f"{'article_id':40} {'n':>4} {'端点覆盖%':>8} {'ev合规%':>8} {'junk%':>6} {'score':>6}"
    print(header)
    for r in out_rows[:15]:
        print(f"{r[0][:38]:40} {r[1]:>4} {r[2]:>8} {r[3]:>8} {r[4]:>6} {r[5]:>6}")
    print(f"... (共 {len(out_rows)} 篇；完整结果见 /tmp/quality_by_article.csv)")

    with open("/tmp/quality_by_article.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["article_id", "n_chunks", "endpoint_coverage_pct",
                    "evidence_compliance_pct", "junk_pct", "quality_score"])
        w.writerows(out_rows)


if __name__ == "__main__":
    main()

