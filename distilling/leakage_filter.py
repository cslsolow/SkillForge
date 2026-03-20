"""
Leakage-prevention filter for experience JSONL files.

When building a retrieval corpus for evaluation, experiences extracted from
the same instances being evaluated must not appear in the corpus (golden-patch
leakage).  This script reads a directory of instance folders (used as an
exclude list), then copies only the non-excluded records from a source
experience directory into a destination directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_EXPERIENCE_FILES = ("keypoints.jsonl", "env_knowledge.jsonl")


def filter_and_append(
    exclude_dir: str | Path,
    src_dir: str | Path,
    dst_dir: str | Path,
) -> None:
    """
    Parameters
    ----------
    exclude_dir : directory whose sub-folder names are instance IDs to exclude.
    src_dir     : source directory containing keypoints.jsonl / env_knowledge.jsonl.
    dst_dir     : destination directory; filtered records are *appended* to any
                  existing files there.
    """
    exclude_dir = Path(exclude_dir)
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    exclude_ids: set[str] = set()
    if exclude_dir.exists():
        exclude_ids = {d.name for d in exclude_dir.iterdir() if d.is_dir()}
    print(f"Loaded {len(exclude_ids)} instance IDs to exclude from {exclude_dir}.")

    for filename in _EXPERIENCE_FILES:
        src_file = src_dir / filename
        dst_file = dst_dir / filename

        if not src_file.exists():
            print(f"Source file {src_file} does not exist, skipping.")
            continue

        print(f"Processing {src_file} -> {dst_file} ...")

        filtered_lines: list[str] = []
        with src_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("instance_id") not in exclude_ids:
                        filtered_lines.append(line if line.endswith("\n") else line + "\n")
                except json.JSONDecodeError:
                    continue

        print(f"  {len(filtered_lines)} records passed the filter.")

        if filtered_lines:
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            with dst_file.open("a", encoding="utf-8") as fh:
                fh.writelines(filtered_lines)
            print(f"  Appended {len(filtered_lines)} records to {dst_file}.")
        else:
            print(f"  No records to append for {filename}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter per-instance experience records to prevent golden-patch leakage, "
            "then append surviving records to a destination directory."
        )
    )
    parser.add_argument(
        "--exclude-dir", required=True,
        help="Directory whose sub-folder names are instance IDs to exclude.",
    )
    parser.add_argument(
        "--src-dir", required=True,
        help="Source directory containing keypoints.jsonl / env_knowledge.jsonl.",
    )
    parser.add_argument(
        "--dst-dir", required=True,
        help="Destination directory to append filtered records into.",
    )
    args = parser.parse_args()
    filter_and_append(args.exclude_dir, args.src_dir, args.dst_dir)


if __name__ == "__main__":
    main()
