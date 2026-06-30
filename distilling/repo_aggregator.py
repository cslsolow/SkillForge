"""
Aggregate per-instance JSONL outputs into repo-level JSONL outputs.

Input (one line per instance):
  keypoints.jsonl     {"instance_id": "...", "kind": "keypoints",    "items": [...]}
  env_knowledge.jsonl {"instance_id": "...", "kind": "env_knowledge","items": [...]}

Output (one line per repo):
  repo_keypoints.jsonl     {"repo_id": "...", "kind": "keypoints",    "items": [...]}
  repo_env_knowledge.jsonl {"repo_id": "...", "kind": "env_knowledge","items": [...]}
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_INSTANCE_RE = re.compile(r"^(?P<repo>.+)-(?P<num>\d+)$")


def repo_id_from_instance_id(instance_id: str) -> str:
    match = _INSTANCE_RE.match(instance_id.strip())
    return match.group("repo") if match else instance_id.strip()


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


@dataclass
class RepoAgg:
    repo_id: str
    env_items: list[dict[str, Any]] = field(default_factory=list)
    keypoint_items: list[dict[str, Any]] = field(default_factory=list)


def aggregate_repo(
    *,
    keypoints_in: Path | None,
    env_in: Path | None,
) -> dict[str, RepoAgg]:
    aggs: dict[str, RepoAgg] = {}

    def ensure(repo_id: str) -> RepoAgg:
        if repo_id not in aggs:
            aggs[repo_id] = RepoAgg(repo_id=repo_id)
        return aggs[repo_id]

    if keypoints_in is not None:
        for rec in _iter_jsonl(keypoints_in):
            instance_id = rec.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                continue
            agg = ensure(repo_id_from_instance_id(instance_id))
            items = rec.get("items", [])
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        it["source_instance_id"] = instance_id
                        agg.keypoint_items.append(it)

    if env_in is not None:
        for rec in _iter_jsonl(env_in):
            instance_id = rec.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                continue
            agg = ensure(repo_id_from_instance_id(instance_id))
            items = rec.get("items", [])
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict):
                        it["source_instance_id"] = instance_id
                        agg.env_items.append(it)

    return aggs


def _write_repo_jsonl(path: Path, lines: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate per-instance experience records into repo-level JSONL."
    )
    parser.add_argument("--keypoints-in", "--local-skills-in", default=None)
    parser.add_argument("--env-in", "--global-skills-in", default=None)
    parser.add_argument("--keypoints-out", "--local-skills-out", default=None)
    parser.add_argument("--env-out", "--global-skills-out", default=None)
    args = parser.parse_args()

    keypoints_in = Path(args.keypoints_in) if args.keypoints_in else None
    env_in = Path(args.env_in) if args.env_in else None
    aggs = aggregate_repo(keypoints_in=keypoints_in, env_in=env_in)

    if args.keypoints_out:
        _write_repo_jsonl(
            Path(args.keypoints_out),
            [
                {"repo_id": repo_id, "kind": "keypoints", "items": agg.keypoint_items}
                for repo_id, agg in sorted(aggs.items())
            ],
        )

    if args.env_out:
        _write_repo_jsonl(
            Path(args.env_out),
            [
                {"repo_id": repo_id, "kind": "env_knowledge", "items": agg.env_items}
                for repo_id, agg in sorted(aggs.items())
            ],
        )


if __name__ == "__main__":
    main()
