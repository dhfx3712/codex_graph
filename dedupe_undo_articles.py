#!/usr/bin/env python3
"""清理 articles/undo 中已在 articles/do 里处理过的文件。"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DO_DIR = ROOT / "articles" / "do"
UNDO_DIR = ROOT / "articles" / "undo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 articles/undo 中的文件是否已存在于 articles/do；若存在则提示并删除。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将删除的文件，不执行实际删除",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not UNDO_DIR.is_dir():
        print(f"目录不存在：{UNDO_DIR}")
        return 2
    if not DO_DIR.is_dir():
        print(f"目录不存在：{DO_DIR}")
        return 2

    processed_names = {path.name for path in DO_DIR.iterdir() if path.is_file()}
    undo_files = sorted(path for path in UNDO_DIR.iterdir() if path.is_file())

    deleted_count = 0
    for undo_path in undo_files:
        if undo_path.name in processed_names:
            rel_path = undo_path.relative_to(ROOT)
            if args.dry_run:
                print(f"[dry-run] 已处理，建议删除：{rel_path}")
            else:
                undo_path.unlink()
                print(f"已处理，删除：{rel_path}")
            deleted_count += 1

    if args.dry_run:
        action_message = f"未执行删除，{deleted_count} 个待删除"
    else:
        action_message = f"已删除 {deleted_count} 个"

    print(
        f"检查完成：undo 共 {len(undo_files)} 个文件，"
        f"重复 {deleted_count} 个，{action_message}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
