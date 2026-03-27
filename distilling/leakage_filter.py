"""
Leakage-prevention filter for experience JSONL files.

When building a retrieval corpus for evaluation, experiences extracted from
the same instances being evaluated must not appear in the corpus (golden-patch
leakage).  This script reads a directory of instance folders (used as an
exclude list), then copies only the non-excluded records from a source
experience directory into a destination directory.

Filtering rules:
1. Exclude experiences from the same instance_id as evaluation instances
2. Exclude experiences from instances with higher numbers in the same repo
   (temporal leakage: synthetic issues created after real issues)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_EXPERIENCE_FILES = ("repo_keypoints.jsonl", "repo_env_knowledge.jsonl")
_INSTANCE_RE = re.compile(r"^(?P<repo>.+)-(?P<num>\d+)$")


def parse_instance_id(instance_id: str) -> tuple[str, int]:
    """Parse instance_id into (repo_id, number). Returns ('', 0) if invalid."""
    match = _INSTANCE_RE.match(instance_id.strip())
    if match:
        return match.group("repo"), int(match.group("num"))
    return instance_id.strip(), 0


def filter_and_append(
    exclude_dir: str | Path,
    src_dir: str | Path,
    dst_dir: str | Path,
) -> None:
    """
    Parameters
    ----------
    exclude_dir : directory whose sub-folder names are instance IDs to exclude.
    src_dir     : source directory containing repo_keypoints.jsonl / repo_env_knowledge.jsonl.
    dst_dir     : destination directory; filtered records are *appended* to any
                  existing files there.
    """
    exclude_dir = Path(exclude_dir)
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    exclude_ids: set[str] = set()
    exclude_map: dict[str, int] = {}
    if exclude_dir.exists():
        for d in exclude_dir.iterdir():
            if d.is_dir():
                exclude_ids.add(d.name)
                repo_id, num = parse_instance_id(d.name)
                if num > 0:
                    exclude_map[repo_id] = min(exclude_map.get(repo_id, float('inf')), num)

    print(f"Loaded {len(exclude_ids)} instance IDs to exclude from {exclude_dir}.")
    print(f"Temporal filter: {len(exclude_map)} repos with min instance numbers.")

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
                    items = data.get("items", [])
                    if not isinstance(items, list):
                        continue

                    filtered_items = []
                    for item in items:
                        if not isinstance(item, dict):
                            continue

                        source_id = item.get("source_instance_id", "")
                        if source_id in exclude_ids:
                            continue

                        orig_id = item.get("original_instance_id", source_id)
                        orig_repo, orig_num = parse_instance_id(orig_id)
                        if orig_repo in exclude_map and orig_num > exclude_map[orig_repo]:
                            continue

                        filtered_items.append(item)

                    if filtered_items:
                        data["items"] = filtered_items
                        filtered_lines.append(json.dumps(data, ensure_ascii=False) + "\n")

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
            "Filter per-instance experience records to prevent golden-patch leakage "
            "and temporal leakage, then append surviving records to a destination directory."
        )
    )
    parser.add_argument(
        "--exclude-dir", required=True,
        help="Directory whose sub-folder names are instance IDs to exclude.",
    )
    parser.add_argument(
        "--src-dir", required=True,
        help="Source directory containing repo_keypoints.jsonl / repo_env_knowledge.jsonl.",
    )
    parser.add_argument(
        "--dst-dir", required=True,
        help="Destination directory to append filtered records into.",
    )
    args = parser.parse_args()
    filter_and_append(args.exclude_dir, args.src_dir, args.dst_dir)


if __name__ == "__main__":
    main()
