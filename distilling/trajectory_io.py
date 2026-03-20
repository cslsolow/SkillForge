"""
I/O helpers for loading agent trajectories and golden-patch metadata.

Trajectory files are the *.traj.json produced by SWE-agent runs.
Leaderboard files are the per-instance JSONL or JSON files that carry the
problem statement, golden patch, and test patch for each SWE-bench instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .schema import GoldenInfo, Trajectory, TrajectoryStep


def _parse_assistant_message(content: str) -> tuple[str, str]:
    thought = content
    action = ""
    marker = "```bash"
    start = content.find(marker)
    if start != -1:
        thought = content[:start].strip()
        rest = content[start + len(marker):]
        end = rest.find("```")
        action = rest[:end].strip() if end != -1 else rest.strip()
    return thought, action


def load_trajectory(path: str | Path) -> Trajectory:
    """Load a *.traj.json file and reconstruct the step sequence."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    info = data.get("info", {})
    messages: List[Dict[str, Any]] = data.get("messages", [])

    steps: List[TrajectoryStep] = []
    step_index = 0

    for i in range(len(messages) - 1):
        msg = messages[i]
        nxt = messages[i + 1]
        if msg.get("role") != "assistant":
            continue
        if nxt.get("role") != "user":
            continue
        content_user = nxt.get("content", "") or ""
        if "<returncode>" not in content_user:
            continue
        thought, action = _parse_assistant_message(msg.get("content", "") or "")
        steps.append(
            TrajectoryStep(
                index=step_index,
                thought=thought,
                action=action,
                observation=content_user,
            )
        )
        step_index += 1

    instance_id = info.get("instance_id") or info.get("task_id")
    return Trajectory(info=info, steps=steps, instance_id=instance_id)


def _extract_modified_files_from_diff(diff_text: str) -> list[str]:
    files: list[str] = []
    if not diff_text:
        return files
    for line in diff_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[2].startswith("a/") and parts[3].startswith("b/"):
            files.append(parts[3][2:])
    return files


def load_golden_info(leaderboard_path: str | Path, instance_id: str) -> GoldenInfo:
    """Load golden patch metadata for a single instance from a leaderboard file."""
    lp = Path(leaderboard_path)
    with lp.open("r", encoding="utf-8") as f:
        leaderboard: Any = json.load(f)

    entry: Dict[str, Any] | None = None
    if isinstance(leaderboard, dict):
        if instance_id in leaderboard and isinstance(leaderboard[instance_id], dict):
            entry = leaderboard[instance_id]
    elif isinstance(leaderboard, list):
        for item in leaderboard:
            if isinstance(item, dict) and item.get("instance_id") == instance_id:
                entry = item
                break

    if entry is None:
        raise KeyError(f"instance_id '{instance_id}' not found in leaderboard")

    patch = entry.get("patch", "") or entry.get("golden_patch", "") or ""
    test_patch = entry.get("test_patch", "") or ""
    problem_statement = entry.get("problem_statement", "") or ""

    solved: bool | None = None
    result = entry.get("result")
    if isinstance(result, str):
        lowered = result.lower()
        if lowered in {"pass", "solved", "success"}:
            solved = True
        elif lowered in {"fail", "failed", "error"}:
            solved = False

    modified_files = (
        _extract_modified_files_from_diff(patch)
        + _extract_modified_files_from_diff(test_patch)
    )

    return GoldenInfo(
        instance_id=instance_id,
        patch=patch,
        test_patch=test_patch,
        problem_statement=problem_statement,
        solved=solved,
        raw=entry,
        modified_files=modified_files,
    )
