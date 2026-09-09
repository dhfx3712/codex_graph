#!/usr/bin/env python3
"""Build a FAISS index from chunks.csv summary_json.

Embedding text is: summary_json.human_summary + "\n" + summary_json.core_argument.
The index uses faiss.IndexIDMap2(IndexFlatIP(384)) and is rebuilt from scratch
every run. Metadata is written to a SQLite file alongside the FAISS index.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CHUNKS_CSV = RESULTS_DIR / "chunks.csv"
FAISS_INDEX_PATH = RESULTS_DIR / "chunks.faiss"
SQLITE_INDEX_PATH = RESULTS_DIR / "chunks_meta.sqlite3"

MODEL_REPO_DIR = Path(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    )
)
DIMENSION = 384
BATCH_SIZE = 16
TEXT_FIELDS = ("human_summary", "core_argument")

LOGGER = logging.getLogger("build_chunk_faiss")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def resolve_model_path() -> Path:
    """Return the local snapshot directory for the cached SentenceTransformer model."""

    ref_path = MODEL_REPO_DIR / "refs" / "main"
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_path = MODEL_REPO_DIR / "snapshots" / revision
            if snapshot_path.is_dir():
                return snapshot_path

    snapshots_dir = MODEL_REPO_DIR / "snapshots"
    if snapshots_dir.is_dir():
        candidates = sorted(
            path for path in snapshots_dir.iterdir() if path.is_dir()
        )
        if candidates:
            return candidates[-1]

    if MODEL_REPO_DIR.is_dir():
        return MODEL_REPO_DIR

    raise FileNotFoundError(f"本地模型目录不存在: {MODEL_REPO_DIR}")


def load_summary_text(raw_summary_json: str) -> str | None:
    """Parse summary_json and return the embedding text, or None if unusable."""

    raw_summary_json = raw_summary_json.strip()
    if not raw_summary_json:
        return None

    try:
        data = json.loads(raw_summary_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    parts: list[str] = []
    for field in TEXT_FIELDS:
        value = data.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            LOGGER.warning(
                "summary_json.%s 不是字符串，跳过该字段: %r",
                field,
                value,
            )
            continue
        stripped = value.strip()
        if stripped:
            parts.append(stripped)

    if not parts:
        return None
    return "\n".join(parts)


def read_chunk_records(max_rows: int | None) -> tuple[list[str], list[dict[str, object]], int]:
    """Read chunks.csv and return texts, faiss-side metadata, and skipped count."""

    if not CHUNKS_CSV.is_file():
        raise FileNotFoundError(f"找不到输入文件: {CHUNKS_CSV}")

    texts: list[str] = []
    metadata_rows: list[dict[str, object]] = []
    skipped = 0

    with CHUNKS_CSV.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError(f"{CHUNKS_CSV} 没有有效表头")

        required_columns = {"id", "article_id", "chunk_index", "summary_json"}
        missing_columns = required_columns - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"{CHUNKS_CSV} 缺少必要列: {', '.join(sorted(missing_columns))}"
            )

        for line_number, row in enumerate(reader, start=2):
            if max_rows is not None and line_number > max_rows + 1:
                break

            chunk_id = (row.get("id") or "").strip()
            article_id = (row.get("article_id") or "").strip()
            raw_chunk_index = (row.get("chunk_index") or "").strip()
            raw_summary_json = row.get("summary_json") or ""

            if not chunk_id or not article_id or not raw_chunk_index:
                LOGGER.warning(
                    "第 %s 行缺少 id/article_id/chunk_index，跳过",
                    line_number,
                )
                skipped += 1
                continue

            try:
                chunk_index = int(raw_chunk_index)
            except ValueError:
                LOGGER.warning(
                    "第 %s 行 chunk_index 不是整数: %r，跳过",
                    line_number,
                    raw_chunk_index,
                )
                skipped += 1
                continue

            text = load_summary_text(raw_summary_json)
            if text is None:
                LOGGER.warning(
                    "第 %s 行 summary_json 无效/无 embedding 文本，跳过",
                    line_number,
                )
                skipped += 1
                continue

            texts.append(text)
            metadata_rows.append(
                {
                    "id": chunk_id,
                    "article_id": article_id,
                    "chunk_index": chunk_index,
                }
            )

    LOGGER.info("读取到 %d 条有效记录，跳过 %d 条", len(texts), skipped)
    return texts, metadata_rows, skipped


def build_index(vectors: np.ndarray) -> faiss.IndexIDMap2:
    """Create an IndexIDMap2(IndexFlatIP) and add normalized vectors."""

    if vectors.ndim != 2 or vectors.shape[1] != DIMENSION:
        raise ValueError(
            f"向量 shape 必须为 (n, {DIMENSION})，实际为 {vectors.shape}"
        )

    index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIMENSION))
    ids = np.arange(vectors.shape[0], dtype=np.int64)
    index.add_with_ids(vectors.astype(np.float32, copy=False), ids)
    return index


def write_sqlite_metadata(
    db_path: Path,
    rows: list[dict[str, object]],
    *,
    model_path: str,
    row_count: int,
    skipped_count: int,
) -> None:
    """Write metadata to an SQLite temp file."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute(
            """
            CREATE TABLE chunk_vectors (
                faiss_id INTEGER PRIMARY KEY,
                id TEXT NOT NULL,
                article_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX idx_chunk_vectors_article ON chunk_vectors(article_id, chunk_index)"
        )
        connection.execute(
            "CREATE INDEX idx_chunk_vectors_id ON chunk_vectors(id)"
        )
        connection.execute(
            """
            CREATE TABLE build_info (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        connection.executemany(
            """
            INSERT INTO chunk_vectors(faiss_id, id, article_id, chunk_index)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    faiss_id,
                    row["id"],
                    row["article_id"],
                    row["chunk_index"],
                )
                for faiss_id, row in enumerate(rows)
            ],
        )

        build_info = {
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_path": model_path,
            "dimension": str(DIMENSION),
            "row_count": str(row_count),
            "skipped_count": str(skipped_count),
            "text_fields": ",".join(TEXT_FIELDS),
            "metric": "inner_product_normalized",
        }
        connection.executemany(
            "INSERT INTO build_info(key, value) VALUES (?, ?)",
            build_info.items(),
        )
        connection.commit()
    finally:
        connection.close()


def replace_outputs_atomically(tmp_index: Path, tmp_db: Path) -> None:
    """Move temp outputs into place, preserving old DB if index move fails."""

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    db_backup = SQLITE_INDEX_PATH.with_name(
        f".{SQLITE_INDEX_PATH.name}.bak-{run_id}"
    )
    had_old_db = SQLITE_INDEX_PATH.exists()

    if had_old_db:
        SQLITE_INDEX_PATH.replace(db_backup)

    try:
        tmp_db.replace(SQLITE_INDEX_PATH)
    except Exception:
        if had_old_db:
            db_backup.replace(SQLITE_INDEX_PATH)
        raise

    try:
        tmp_index.replace(FAISS_INDEX_PATH)
    except Exception:
        if SQLITE_INDEX_PATH.exists():
            SQLITE_INDEX_PATH.unlink()
        if had_old_db:
            db_backup.replace(SQLITE_INDEX_PATH)
        raise

    if had_old_db:
        db_backup.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FAISS index from results/chunks.csv"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="只处理前 N 行 CSV 数据，用于本地小规模测试",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path = resolve_model_path()
    LOGGER.info("加载本地模型: %s", model_path)
    model = SentenceTransformer(str(model_path), device="cpu")

    texts, metadata_rows, skipped = read_chunk_records(args.max_rows)
    if not texts:
        raise RuntimeError(
            "没有可写入 FAISS 的有效记录；旧的索引文件保持不动。"
        )

    LOGGER.info("开始 encode，batch_size=%s, normalize=True", BATCH_SIZE)
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    LOGGER.info("向量 shape: %s", vectors.shape)

    index = build_index(vectors)
    LOGGER.info("FAISS index 大小: %s", index.ntotal)

    run_id = uuid.uuid4().hex
    tmp_index = FAISS_INDEX_PATH.with_name(
        f".{FAISS_INDEX_PATH.name}.{run_id}.tmp"
    )
    tmp_db = SQLITE_INDEX_PATH.with_name(
        f".{SQLITE_INDEX_PATH.name}.{run_id}.tmp"
    )

    try:
        LOGGER.info("写入临时 FAISS 文件: %s", tmp_index)
        faiss.write_index(index, str(tmp_index))
        LOGGER.info("写入临时 SQLite 文件: %s", tmp_db)
        write_sqlite_metadata(
            tmp_db,
            metadata_rows,
            model_path=str(model_path),
            row_count=len(metadata_rows),
            skipped_count=skipped,
        )
        LOGGER.info("原子替换最终输出文件")
        replace_outputs_atomically(tmp_index, tmp_db)
    finally:
        for path in (tmp_index, tmp_db):
            if path.exists():
                path.unlink(missing_ok=True)

    LOGGER.info(
        "完成: %s / %s，共写入 %d 条向量",
        FAISS_INDEX_PATH,
        SQLITE_INDEX_PATH,
        len(metadata_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
