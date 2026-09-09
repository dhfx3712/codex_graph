#!/usr/bin/env python3
"""Query the chunk FAISS index built by build_chunk_faiss.py.

Query embedding must use the same local model, normalization, and metric as
the index build: SentenceTransformer(...), normalize_embeddings=True,
IndexFlatIP.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from build_chunk_faiss import (
    BATCH_SIZE,
    DIMENSION,
    FAISS_INDEX_PATH,
    SQLITE_INDEX_PATH,
    resolve_model_path,
)


def load_model() -> SentenceTransformer:
    path = resolve_model_path()
    return SentenceTransformer(str(path), device="cpu")


def search(query: str, top_k: int) -> list[dict[str, object]]:
    if not FAISS_INDEX_PATH.is_file():
        raise FileNotFoundError(f"找不到 FAISS 文件: {FAISS_INDEX_PATH}")
    if not SQLITE_INDEX_PATH.is_file():
        raise FileNotFoundError(f"找不到 SQLite 文件: {SQLITE_INDEX_PATH}")

    model = load_model()
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    if index.d != DIMENSION:
        raise ValueError(f"FAISS 维度不是 {DIMENSION}: {index.d}")

    top_k = min(top_k, index.ntotal)
    if top_k <= 0:
        return []

    vector = model.encode(
        [query],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
        device="cpu",
    )
    vector = np.asarray(vector, dtype=np.float32)

    distances, indices = index.search(vector, top_k)

    connection = sqlite3.connect(SQLITE_INDEX_PATH)
    try:
        placeholders = ",".join("?" for _ in indices[0])
        cursor = connection.execute(
            f"""
            SELECT faiss_id, id, article_id, chunk_index
            FROM chunk_vectors
            WHERE faiss_id IN ({placeholders})
            """,
            [int(faiss_id) for faiss_id in indices[0]],
        )
        metadata_by_id = {
            row[0]: {
                "id": row[1],
                "article_id": row[2],
                "chunk_index": row[3],
            }
            for row in cursor.fetchall()
        }
    finally:
        connection.close()

    results: list[dict[str, object]] = []
    for faiss_id, score in zip(indices[0], distances[0]):
        if int(faiss_id) not in metadata_by_id:
            continue
        results.append(
            {
                "faiss_id": int(faiss_id),
                "score": float(score),
                **metadata_by_id[int(faiss_id)],
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query results/chunks.faiss and return SQLite-backed chunk locations."
    )
    parser.add_argument("query", help="自然语言查询，例如：美债规模")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="返回的最大结果数，默认 5",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    hits = search(args.query, args.top_k)
    print(json.dumps({"query": args.query, "hits": hits}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
