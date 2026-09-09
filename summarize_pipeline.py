#!/usr/bin/env python3
"""Process markdown articles in articles/undo, emit summaries and archive to articles/do."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
ARTICLES_DIR = BASE_DIR / "articles"
UNDO_DIR = ARTICLES_DIR / "undo"
DO_DIR = ARTICLES_DIR / "do"
RESULTS_DIR = BASE_DIR / "results"
OUT_ARTICLES = RESULTS_DIR / "articles.csv"
OUT_CHUNKS = RESULTS_DIR / "chunks.csv"

ARTICLE_FIELDS = [
    "id",
    "title",
    "type",
    "source",
    "author",
    "raw_content",
    "summary",
    "summary_json",
    "word_count",
    "status",
    "created_at",
    "updated_at",
]

CHUNK_FIELDS = [
    "id",
    "article_id",
    "chunk_index",
    "content",
    "summary",
    "summary_json",
    "token_count",
    "embedding",
    "created_at",
]

ENV_FILE = Path("/home/ubuntu/codex_english/.env")
VOLC_BASE_URL = os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/coding/v3")
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")

CHUNK_TARGET = 700
CHUNK_MAX = 1000
DEFAULT_MAX_WORKERS = 4


ARTICLE_SUMMARY_PROMPT = """你是一位专业分析师，同时为知识图谱构建系统生成可解析数据。

文章标题：<<TITLE>>
文章内容：
<<CONTENT>>

请完成两部分：
1. human_summary：适合人阅读的 300-500 字结构化摘要，突出核心主题、关键事件、因果关系和重要数字。
2. graph_data：适合程序解析，只基于原文，不得添加原文没有的内容。所有实体和关系都必须能找到原文证据。

实体要求：
- 提取 8-15 个重要实体。
- 每个实体字段：name（原始名称）、type（国家/机构/政策工具/事件/概念/资产/人物/制度等）、aliases（可能有多个别名，至少 1 个）、evidence（原文短语或短句）。
- 同名实体只出现一次，不同写法放进 aliases。

关系要求：
- 提取 8-15 条实体关系。
- 每个关系字段：source、target、relation、evidence。
- source 和 target 必须出现在 entities 的 name 或 aliases 中。
- relation 只能从以下集合选择：导致、服务、依赖、配合、削弱、构成、属于、决定。

输出要求：
- 只输出一个合法 JSON 对象，不要 Markdown 代码块，不要额外解释。
- JSON 顶层字段必须为：human_summary、core_theme、key_points、entities、relationships。
- key_points 为 3-5 条字符串。
- entities 和 relationships 必须是数组。
- 如果原文较短导致数量不足，可以低于下限，但 human_summary 中需说明信息有限。

输出示例：
{
  "human_summary": "本文……",
  "core_theme": "……",
  "key_points": ["……", "……"],
  "entities": [
    {"name": "美债", "type": "金融工具", "aliases": ["美国国债"], "evidence": "美债的本质就是美国的联邦政府……"}
  ],
  "relationships": [
    {"source": "美债", "target": "美元霸权", "relation": "服务", "evidence": "核心目标是维护美元霸权"}
  ]
}
"""


CHUNK_SUMMARY_PROMPT = """你是一位专业文本分析师，为一个长文章切片生成适合人阅读的摘要和适合程序解析的知识图谱数据。

切片上下文：
- 所属文章：<<TITLE>>
- 切片序号：第 <<INDEX>> 个切片，共 <<TOTAL>> 个切片
- 本切片文本：
<<CONTENT>>
- 前一切片末尾（用于衔接，可能为空）：
<<PREVIOUS>>
- 后一切片开头（用于衔接，可能为空）：
<<NEXT>>

请完成两部分：
1. human_summary：100-150 字自然语言摘要，概括本切片核心论点，保留数字、日期、名称。如果开头或结尾被截断，请在摘要中说明。
2. graph_data：适合程序解析，只基于本切片内容，不得使用前后切片内容作为证据。

实体和关系要求：
- 提取 3-8 个本切片关键实体，字段为 name、type、aliases、evidence。
- 提取 2-6 条本切片关键关系，字段为 source、target、relation、evidence。
- source 和 target 必须出现在 entities 的 name 或 aliases 中。
- relation 只能从以下集合选择：导致、服务、依赖、配合、削弱、构成、属于、决定。

输出要求：
- 只输出一个合法 JSON 对象，不要 Markdown 代码块，不要额外解释。
- JSON 顶层字段必须为：human_summary、core_argument、entities、relationships、context_to_next、boundary_incomplete。
- context_to_next 用一句话说明本切片与下一个切片的衔接线索，如果后面切片为空则写空字符串。
- boundary_incomplete 为布尔值，表示本切片开头或结尾是否被截断。

输出示例：
{
  "human_summary": "本切片介绍……",
  "core_argument": "……",
  "entities": [
    {"name": "美债", "type": "金融工具", "aliases": ["美国国债"], "evidence": "美债的本质……"}
  ],
  "relationships": [
    {"source": "财政部", "target": "美债", "relation": "导致", "evidence": "为了弥补财政赤字……"}
  ],
  "context_to_next": "下一段将从历史起源展开",
  "boundary_incomplete": false
}
"""


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def ensure_directories() -> None:
    ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    UNDO_DIR.mkdir(parents=True, exist_ok=True)
    DO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def pending_markdown_files() -> list[Path]:
    ensure_directories()
    return sorted(
        path
        for path in UNDO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".md"
    )


def load_existing_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            return set()
        return {row["id"] for row in reader if row.get("id")}


def ensure_header(csv_path: Path, fieldnames: list[str]) -> None:
    if csv_path.exists():
        return
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()


def append_unique_rows(
    csv_path: Path,
    fieldnames: list[str],
    rows: list[dict],
    seen_ids: set[str],
) -> list[dict]:
    new_rows = [row for row in rows if row["id"] not in seen_ids]
    if not new_rows:
        return []

    with csv_path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writerows(new_rows)

    for row in new_rows:
        seen_ids.add(row["id"])
    return new_rows


def move_to_done(md_path: Path) -> Path:
    destination = DO_DIR / md_path.name
    if not destination.exists():
        md_path.rename(destination)
        return destination

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    candidate = DO_DIR / f"{md_path.stem}_{timestamp}{md_path.suffix}"
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = DO_DIR / f"{md_path.stem}_{timestamp}_{counter}{md_path.suffix}"

    md_path.rename(candidate)
    return candidate


def clean_markdown(text: str) -> str:
    text = re.sub(r"```[a-zA-Z0-9_+-]*", "", text)
    text = re.sub(r"```", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [part.strip() for part in parts if part.strip()]


def split_long_sentence(sentence: str, max_len: int) -> list[str]:
    """Keep most chunks in the 500-1000 range, including single overlong sentences."""
    if len(sentence) <= max_len:
        return [sentence]
    pieces: list[str] = []
    clauses = re.split(r"(?<=[，,：:、])", sentence)
    current = ""
    for clause in clauses:
        if clause.strip() == "":
            continue
        if len(clause) > max_len:
            if current:
                pieces.append(current)
                current = ""
            for start in range(0, len(clause), max_len):
                pieces.append(clause[start:start + max_len].strip())
            continue
        if len(current) + len(clause) <= max_len:
            current += clause
        else:
            if current:
                pieces.append(current)
            current = clause
    if current:
        pieces.append(current)
    return pieces or [sentence[:max_len], sentence[max_len:]]


def chunk_article(text: str, target: int = CHUNK_TARGET, max_len: int = CHUNK_MAX) -> list[str]:
    sentences = split_sentences(text)
    fragments: list[str] = []
    for sentence in sentences:
        fragments.extend(split_long_sentence(sentence, max_len))

    chunks: list[str] = []
    current = ""
    for fragment in fragments:
        fragment = fragment.strip()
        if not fragment:
            continue
        if not current:
            current = fragment
            continue
        if len(current) >= target or len(current) + len(fragment) > max_len:
            chunks.append(current.strip())
            current = fragment
        else:
            current += fragment

    if current.strip():
        chunks.append(current.strip())

    # Merge any unusually short chunk into the previous chunk when possible.
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < target // 2 and i > 0:
            if len(chunks[i - 1]) + len(chunks[i]) <= max_len:
                chunks[i - 1] += chunks[i]
                chunks.pop(i)
                continue
        i += 1

    return chunks


def call_llm(
    prompt: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool = False,
) -> str:
    load_env_file()
    api_key = os.getenv("ARK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ARK_API_KEY is not set and /home/ubuntu/codex_english/.env was not found"
        )
    url = f"{VOLC_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_json_response(content: str) -> dict:
    text = content.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    text = "\n".join(lines).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"LLM 返回内容不是合法 JSON: {content[:300]}")

    depth = 0
    in_string = False
    escaped = False
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end is None:
        raise ValueError(f"LLM 返回内容没有完整 JSON 对象: {content[:300]}")

    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM 返回的 JSON 顶层不是对象")
    return data


def adjacent_contexts(chunks: list[str], index: int) -> tuple[str, str]:
    previous = chunks[index - 2][-120:] if index > 1 else ""
    following = chunks[index][:120] if index < len(chunks) else ""
    return previous, following


def generate_article_rows(md_path: Path) -> tuple[dict, list[dict]]:
    article_md = md_path.read_text(encoding="utf-8")
    article_text = clean_markdown(article_md)
    title = md_path.stem
    chunks = chunk_article(article_text)

    print(f"文章标题: {title}")
    print(f"清洗后字数: {len(article_text)}")
    print(f"切片数量: {len(chunks)}")

    article_prompt = (
        ARTICLE_SUMMARY_PROMPT.replace("<<TITLE>>", title).replace("<<CONTENT>>", article_text)
    )
    article_raw = call_llm(
        article_prompt,
        max_tokens=2500,
        temperature=0.3,
        json_mode=True,
    )
    article_data = parse_json_response(article_raw)
    article_summary = article_data.get("human_summary") or article_raw
    article_json = json.dumps(article_data, ensure_ascii=False)

    chunk_summaries: dict[int, str] = {}
    chunk_json_by_index: dict[int, str] = {}
    for index, chunk in enumerate(chunks, start=1):
        print(f"正在生成切片摘要 {index}/{len(chunks)}")
        previous_snippet, next_snippet = adjacent_contexts(chunks, index)

        chunk_prompt = (
            CHUNK_SUMMARY_PROMPT
            .replace("<<TITLE>>", title)
            .replace("<<INDEX>>", str(index))
            .replace("<<TOTAL>>", str(len(chunks)))
            .replace("<<CONTENT>>", chunk)
            .replace("<<PREVIOUS>>", previous_snippet)
            .replace("<<NEXT>>", next_snippet)
        )
        chunk_raw = call_llm(
            chunk_prompt,
            max_tokens=1200,
            temperature=0.3,
            json_mode=True,
        )
        chunk_data = parse_json_response(chunk_raw)
        chunk_summaries[index] = chunk_data.get("human_summary") or chunk_raw
        chunk_json_by_index[index] = json.dumps(chunk_data, ensure_ascii=False)

    article_id = str(uuid.uuid5(uuid.NAMESPACE_URL, md_path.resolve().as_uri()))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    word_count = len(re.sub(r"\s+", "", article_text))

    article_row = {
        "id": article_id,
        "title": title,
        "type": "",
        "source": "本地文档",
        "author": "未标注",
        "raw_content": article_md.strip(),
        "summary": article_summary,
        "summary_json": article_json,
        "word_count": word_count,
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }

    chunk_rows = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_rows.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{article_id}:{index}")),
                "article_id": article_id,
                "chunk_index": index,
                "content": chunk,
                "summary": chunk_summaries[index],
                "summary_json": chunk_json_by_index[index],
                "token_count": len(chunk),
                "embedding": "",
                "created_at": now,
            }
        )

    return article_row, chunk_rows


def main() -> int:
    ensure_directories()
    load_env_file()
    pending_files = pending_markdown_files()

    if not pending_files:
        print(f"{UNDO_DIR} 中没有待处理的 .md 文件")
        return 0

    ensure_header(OUT_ARTICLES, ARTICLE_FIELDS)
    ensure_header(OUT_CHUNKS, CHUNK_FIELDS)
    seen_article_ids = load_existing_ids(OUT_ARTICLES)
    seen_chunk_ids = load_existing_ids(OUT_CHUNKS)

    max_workers = max(
        1,
        int(os.getenv("SUMMARIZE_PIPELINE_WORKERS", str(DEFAULT_MAX_WORKERS))),
    )
    print(f"使用 {max_workers} 个线程处理 {len(pending_files)} 个文件")

    failed_files: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(generate_article_rows, md_path): md_path
            for md_path in pending_files
        }

        for future in as_completed(future_to_path):
            md_path = future_to_path[future]
            try:
                article_row, chunk_rows = future.result()
                written_articles = append_unique_rows(
                    OUT_ARTICLES,
                    ARTICLE_FIELDS,
                    [article_row],
                    seen_article_ids,
                )
                written_chunks = append_unique_rows(
                    OUT_CHUNKS,
                    CHUNK_FIELDS,
                    chunk_rows,
                    seen_chunk_ids,
                )
                print(
                    f"\n处理完成: {md_path.name}, "
                    f"新增文章 {len(written_articles)} 行, "
                    f"新增切片 {len(written_chunks)} 行"
                )

                moved_path = move_to_done(md_path)
                print(f"已归档: {moved_path.relative_to(BASE_DIR)}")
            except Exception as exc:
                failed_files.append(md_path.name)
                print(f"处理失败: {md_path.name}: {exc}")


    if failed_files:
        print("\n以下文件处理失败，保留在 undo:")
        for name in failed_files:
            print(f"- {name}")
        return 1

    print("\n全部处理完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
