"""Experience loading and formatting utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_experience_jsonl(path: Path | str) -> dict[str, dict]:
    """Load experience JSONL file and return a mapping from instance_id to record."""
    path = Path(path)
    if not path.exists():
        return {}
    mapping: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        instance_id = record.get("instance_id")
        if instance_id:
            mapping[instance_id] = record
    return mapping


_INSTANCE_RE = re.compile(r"^(?P<repo>.+)-(?P<num>\d+)$")
_TIMESTAMP_RE = re.compile(r"^\d{10,}$")  # Pattern for long timestamps

def repo_id_from_instance_id(instance_id: str) -> str:
    match = _INSTANCE_RE.match(instance_id.strip())
    return match.group("repo") if match else instance_id.strip()


def num_from_instance_id(instance_id: str) -> int:
    """Extract the numeric suffix from an instance_id (e.g., 'repo-1234' -> 1234)."""
    match = _INSTANCE_RE.match(instance_id.strip())
    if match:
        try:
            return int(match.group("num"))
        except ValueError:
            pass
    return 0


def load_domain2_experience_jsonl(
    *,
    keypoints_path: Path | str | None = None,
    env_knowledge_path: Path | str | None = None,
) -> dict[str, dict]:
    """
    Load domain2 JSONL outputs and merge them into a single mapping keyed by either:
    - instance_id (per-instance files), or
    - repo_id (repo-level aggregated files).

    Mapping value schema:
      {"kind": "domain2", "keypoints": [...], "env_knowledge": [...], "instance_id"/"repo_id": ...}
    """
    mapping: dict[str, dict] = {}

    def ensure(key: str, *, instance_id: str | None = None, repo_id: str | None = None) -> dict:
        if key not in mapping:
            record: dict = {"kind": "domain2"}
            if instance_id:
                record["instance_id"] = instance_id
            if repo_id:
                record["repo_id"] = repo_id
            mapping[key] = record
        return mapping[key]

    def read_jsonl_lines(path: Path) -> list[dict]:
        if not path.exists():
            return []
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    if keypoints_path is not None:
        for rec in read_jsonl_lines(Path(keypoints_path)):
            instance_id = rec.get("instance_id")
            repo_id = rec.get("repo_id")
            key = instance_id or repo_id
            if not key:
                continue
            record = ensure(key, instance_id=instance_id, repo_id=repo_id)
            if "keypoints" not in record:
                record["keypoints"] = []
            
            items = rec.get("items", [])
            for item in items:
                # Ensure each item has an instance_id for filtering
                if instance_id and "instance_id" not in item:
                    item["instance_id"] = instance_id
                elif "source_instance_id" in item and "instance_id" not in item:
                    item["instance_id"] = item["source_instance_id"]
            
            record["keypoints"].extend(items)

    if env_knowledge_path is not None:
        for rec in read_jsonl_lines(Path(env_knowledge_path)):
            instance_id = rec.get("instance_id")
            repo_id = rec.get("repo_id")
            key = instance_id or repo_id
            if not key:
                continue
            record = ensure(key, instance_id=instance_id, repo_id=repo_id)
            if "env_knowledge" not in record:
                record["env_knowledge"] = []
            
            items = rec.get("items", [])
            for item in items:
                # Ensure each item has a source_instance_id for filtering
                if instance_id and "source_instance_id" not in item:
                    item["source_instance_id"] = instance_id
            
            record["env_knowledge"].extend(items)

    return mapping


def format_experience(record: dict) -> str:
    """Format an experience record into a string for injection into the agent's context.
    
    Args:
        record: A dict containing experience data, e.g.:
            {
                "instance_id": "...",
                "resolved": True/False,
                "kind": "success_experience" or "failure_reflection",
                "strategies": [...],
                "missing_domain_knowledge": [...],
                ...
            }
    
    Returns:
        A formatted string to be added as a user message.
    """
    kind = record.get("kind", "")
    lines = ["You can refer to these domain-specific knowledge to fix this task:", ""]
    
    if kind == "success_experience":
        strategies = record.get("strategies", [])
        if strategies:
            lines.append("Strategies that worked for similar issues:")
            for i, s in enumerate(strategies, 1):
                lines.append(f"  {i}. {s}")
    elif kind == "failure_reflection":
        missing_knowledge = record.get("missing_domain_knowledge", [])
        if missing_knowledge:
            # lines.append("Domain knowledge needed for similar issues:")
            for item in missing_knowledge:
                if isinstance(item, dict):
                    desc = item.get("description", "")
                    resources = item.get("suggested_resources", "")
                    lines.append(f"  - {desc}")
                    if resources:
                        lines.append(f"    Suggested resources: {resources}")
                else:
                    lines.append(f"  - {item}")
        lines.append(f"Correct approach: {record.get('correct_approach', '')}")
    return "\n".join(lines)
