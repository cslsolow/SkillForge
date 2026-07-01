"""
Per-instance experience extraction pipeline.

Reads agent trajectories and golden-patch metadata, then calls an LLM to
extract two types of experience records per instance:

  keypoints     -- bug-fixing lessons tied to specific API paths
  env_knowledge -- environment/API-usage knowledge tied to specific API paths

Resolution status is taken from a pre-computed eval JSON when available;
otherwise the pipeline asks the LLM to judge patch equivalence.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .access_extractor import extract_all_code_accesses
from .code_index import CodeIndex
from .llm_client import _query_with_retry
from .schema import CodeScope, GoldenInfo, Trajectory
from .source_files import is_source_file, is_test_file
from .trajectory_io import load_trajectory, _extract_modified_files_from_diff


# ---------------------------------------------------------------------------
# LLM call log
# ---------------------------------------------------------------------------


@dataclass
class InstanceCallLog:
    instance_id: str
    resolved: bool
    kind: str
    model: str
    messages: list[dict[str, str]]
    output: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def to_markdown(self) -> str:
        lines: list[str] = [
            f"## {self.instance_id}", "",
            f"- **resolved**: {self.resolved}",
            f"- **kind**: {self.kind}",
            f"- **model**: {self.model}",
            f"- **prompt_tokens**: {self.prompt_tokens}",
            f"- **completion_tokens**: {self.completion_tokens}",
            f"- **total_tokens**: {self.total_tokens}", "",
        ]
        for msg in self.messages:
            lines += [f"### {msg.get('role', 'unknown').upper()}", "", "```",
                      msg.get("content", ""), "```", ""]
        lines += ["### OUTPUT", "", "```", self.output, "```", "", "---", ""]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Leaderboard loading (JSONL / folder-based / JSON list or dict)
# ---------------------------------------------------------------------------

_LEADERBOARD_CACHE: dict[str, dict[str, Any]] = {}


def _load_golden_info(path: Path, instance_id: str) -> GoldenInfo:
    cache_key = str(path)
    cached = _LEADERBOARD_CACHE.get(cache_key, {}).get(instance_id)
    if cached:
        return GoldenInfo(
            instance_id=cached.get("instance_id", instance_id),
            problem_statement=cached.get("problem_statement", ""),
            patch=cached.get("patch", cached.get("golden_patch", "")),
            test_patch=cached.get("test_patch", ""),
            modified_files=cached.get("modified_files", []),
            raw=cached,
        )

    record: dict[str, Any] | None = None
    instance_jsonl = path / instance_id / "instances.jsonl"
    if instance_jsonl.exists():
        record = json.loads(instance_jsonl.read_text(encoding="utf-8").splitlines()[0])
    elif path.is_file() and ("jsonl" in path.name or path.suffix == ".jsonl"):
        _LEADERBOARD_CACHE[cache_key] = {}
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    iid = data.get("instance_id")
                    if iid:
                        _LEADERBOARD_CACHE[cache_key][iid] = data
                except Exception:
                    continue
        record = _LEADERBOARD_CACHE[cache_key].get(instance_id)
    elif path.is_file() and path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        _LEADERBOARD_CACHE[cache_key] = {}
        if isinstance(data, list):
            for item in data:
                iid = item.get("instance_id")
                if iid:
                    _LEADERBOARD_CACHE[cache_key][iid] = item
        elif isinstance(data, dict):
            for iid, info in data.items():
                if isinstance(info, dict):
                    _LEADERBOARD_CACHE[cache_key][iid] = info
                else:
                    _LEADERBOARD_CACHE[cache_key][instance_id] = data
                    break
        record = _LEADERBOARD_CACHE[cache_key].get(instance_id)

    if record is None:
        raise ValueError(f"Instance {instance_id!r} not found in {path}")
    return GoldenInfo(
        instance_id=record.get("instance_id", instance_id),
        problem_statement=record.get("problem_statement", ""),
        patch=record.get("patch", record.get("golden_patch", "")),
        test_patch=record.get("test_patch", ""),
        modified_files=record.get("modified_files", []),
        raw=record,
    )


def _load_eval_ids(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None or not path.exists():
        return set(), set()
    data = json.loads(path.read_text(encoding="utf-8"))
    resolved_ids = set(data.get("resolved_ids", []))
    unresolved_ids = set(data.get("unresolved_ids", []))
    empty_patch_ids = set(data.get("empty_patch_ids", []))
    return resolved_ids, unresolved_ids | empty_patch_ids


def _iter_traj_instances(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        traj_files = list(child.glob(f"{child.name}_trajectory.json"))
        if not traj_files:
            traj_files = list(child.glob("*.traj.json"))
        if not traj_files:
            traj_files = list(child.glob("*_trajectory.json"))
        if traj_files:
            mapping[child.name] = traj_files[0]
    return mapping


# ---------------------------------------------------------------------------
# LLM-based resolution judgement (fallback when eval JSON is absent)
# ---------------------------------------------------------------------------


def _check_if_resolved_by_llm(
    instance_id: str,
    problem_statement: str,
    golden_patch: str,
    submission_patch: str,
    model: str,
    **llm_kwargs: Any,
) -> bool:
    if not submission_patch.strip():
        return False
    system_prompt = (
        "You are an expert software engineer. Determine if a candidate patch correctly fixes "
        "a reported bug by comparing it with a Golden Patch (a known correct fix). "
        "Focus on logical equivalence and root-cause resolution; ignore formatting differences. "
        'Respond with JSON only: {"resolved": true/false, "reason": "..."}'
    )
    user_content = "\n".join([
        f"Instance ID: {instance_id}",
        "\n=== Problem Statement ===", problem_statement,
        "\n=== Golden Patch ===", golden_patch,
        "\n=== Candidate Patch ===", submission_patch,
        "\nReturn JSON only.",
    ])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    try:
        response = _query_with_retry(model, messages, **llm_kwargs)
        content = response.choices[0].message.content or ""
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return bool(json.loads(match.group(0)).get("resolved", False))
    except Exception as exc:
        print(f"Warning: LLM resolution check failed for {instance_id}: {exc}")
    return False


# ---------------------------------------------------------------------------
# Candidate API path building
# ---------------------------------------------------------------------------


def _build_trajectory_text(traj: Trajectory, max_steps: int | None) -> str:
    lines: list[str] = []
    steps = traj.steps if max_steps is None else traj.steps[:max_steps]
    for step in steps:
        step_lines = [f"=== Step {step.index} ==="]
        if step.thought:
            step_lines.append(f"[Thought] {step.thought}")
        if step.action:
            step_lines.append(f"[Action] {step.action}")
        if len(step_lines) > 1:
            lines.extend(step_lines)
            lines.append("")
    return "\n".join(lines)


def _align_accesses_to_scopes(
    traj: Trajectory,
    repo_root: Path | None,
    extra_files: list[str],
) -> tuple[list[CodeScope], dict[str, int]]:
    accesses = extract_all_code_accesses(traj.steps)
    filepath_access_counts: dict[str, int] = {}
    for access in accesses:
        filepath_access_counts[access.filepath] = (
            filepath_access_counts.get(access.filepath, 0) + 1
        )

    candidate_files = {
        fp for fp in (set(filepath_access_counts) | set(extra_files))
        if is_source_file(fp) and not is_test_file(fp)
    }

    code_index = CodeIndex()
    if repo_root is not None:
        for fp in sorted(candidate_files):
            code_index.add_file_from_repo(repo_root, fp)

    scope_counts: dict[str, int] = {}
    scopes: list[CodeScope] = []
    for access in accesses:
        if not is_source_file(access.filepath) or is_test_file(access.filepath):
            continue
        for class_name, func_name in code_index.find_scope_for_range(
            access.filepath, access.start_line, access.end_line
        ):
            scope = CodeScope(
                filepath=access.filepath, class_name=class_name, function_name=func_name
            )
            key = scope.scope_key
            scope_counts[key] = scope_counts.get(key, 0) + 1
            scopes.append(scope)

    for fp in sorted(extra_files):
        if not is_source_file(fp) or is_test_file(fp):
            continue
        scope = CodeScope(filepath=fp)
        key = scope.scope_key
        if key not in scope_counts:
            scope_counts[key] = 0
            scopes.append(scope)

    return scopes, scope_counts


def _build_candidate_api_paths(
    traj: Trajectory,
    golden: GoldenInfo,
    submission_patch: str,
    repo_root: Path | None,
    max_candidates: int,
) -> list[str]:
    extra_files = (
        _extract_modified_files_from_diff(submission_patch)
        + _extract_modified_files_from_diff(golden.patch)
        + _extract_modified_files_from_diff(golden.test_patch)
        + (golden.modified_files or [])
    )
    extra_files = [fp for fp in extra_files if is_source_file(fp) and not is_test_file(fp)]
    scopes, scope_counts = _align_accesses_to_scopes(traj, repo_root, extra_files)

    def sort_key(scope: CodeScope) -> tuple[int, int, str]:
        count = scope_counts.get(scope.scope_key, 0)
        specificity = 2 if scope.function_name else (1 if scope.class_name else 0)
        return (-count, -specificity, scope.scope_key)

    seen: set[str] = set()
    ordered: list[str] = []
    for scope in sorted(scopes, key=sort_key):
        candidate = scope.scope_key
        if candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
        if len(ordered) >= max_candidates:
            break
    return ordered


# ---------------------------------------------------------------------------
# JSON parsing / validation
# ---------------------------------------------------------------------------


def _parse_json_array(content: str) -> list[dict[str, Any]] | None:
    cleaned = content.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _parse_json_dict(content: str) -> dict[str, Any] | None:
    cleaned = content.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None


def _validate_keypoints(
    items: list[dict[str, Any]], candidates: set[str]
) -> list[dict[str, Any]] | None:
    validated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        api_path = item.get("api_path")
        if not isinstance(api_path, str) or api_path not in candidates:
            return None
        exp = item.get("experience_or_reflection")
        thinking = item.get("thinking")
        if not isinstance(exp, str) or not isinstance(thinking, str):
            return None
        validated.append(
            {"api_path": api_path, "experience_or_reflection": exp, "thinking": thinking}
        )
    return validated


def _validate_env_knowledge(
    items: list[dict[str, Any]], candidates: set[str]
) -> list[dict[str, Any]] | None:
    validated: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        api_path = item.get("api_path")
        if not isinstance(api_path, str) or api_path not in candidates:
            return None
        purpose = item.get("purpose")
        if not isinstance(purpose, str):
            return None
        playbook = item.get("playbook", "")
        thinking = item.get("thinking", "")
        if not isinstance(playbook, str) or not isinstance(thinking, str):
            return None
        related_apis = item.get("related_apis", [])
        if not isinstance(related_apis, list):
            return None
        normalized_related: list[dict[str, str]] = []
        for rel in related_apis:
            if isinstance(rel, str):
                if rel not in candidates:
                    return None
                normalized_related.append({"api_path": rel, "reason": ""})
            elif isinstance(rel, dict):
                rel_api = rel.get("api_path")
                rel_reason = rel.get("reason", "")
                if not isinstance(rel_api, str) or rel_api not in candidates:
                    return None
                if not isinstance(rel_reason, str):
                    return None
                normalized_related.append({"api_path": rel_api, "reason": rel_reason})
            else:
                return None
        validated.append({
            "api_path": api_path, "purpose": purpose,
            "playbook": playbook, "related_apis": normalized_related, "thinking": thinking,
        })
    return validated


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_KP_SYSTEM = (
    "You extract keypoints from bug-fixing trajectories.\n"
    "Respond with a valid JSON ARRAY only.\n"
    "Each item must have keys: api_path, experience_or_reflection, thinking.\n"
    "api_path MUST be chosen from the Candidate API Paths list. Do NOT invent APIs.\n"
    "You MAY output multiple items per api_path (max 3).\n"
    "experience_or_reflection: what this API contributed to success/failure (<= 200 chars).\n"
    "If resolved=false, focus on the agent's mistake and what should be done instead.\n"
    "thinking: why this API was selected (<= 120 chars).\n"
)

_ENV_SYSTEM = (
    "You extract environment knowledge from a bug-fixing trajectory.\n"
    "Respond with a valid JSON ARRAY only.\n"
    "Each item must have keys: api_path, purpose, playbook, related_apis, thinking.\n"
    "api_path and related_apis MUST be from the Candidate API Paths list.\n"
    "purpose: what this API does (<= 160 chars).\n"
    "playbook: how to use it in this context (<= 200 chars).\n"
    "related_apis: array of {api_path, reason} objects (reason <= 120 chars).\n"
    "thinking: why this API matters here (<= 120 chars).\n"
)

_ISSUE_FOCUS_SYSTEM = (
    "You summarize a SWE-bench issue into its core requirements for reuse as memory.\n"
    'Respond with a JSON object with exactly these keys: "symptom", "expected_behavior", "constraints".\n'
    "Each value: 1 short sentence, no code blocks or markdown, no verbatim copy.\n"
)


def _build_issue_focus_user_prompt(
    *, problem_statement: str, golden_patch: str, golden_tests: str
) -> str:
    ps = _clean_problem_statement(problem_statement or "")
    return "\n".join([
        "Summarize the issue into a structured JSON object.",
        "Use the golden patch and tests to identify the root requirement.",
        "", "<problem_statement>", ps, "</problem_statement>",
        "", "<test_cases_or_test_patch>", golden_tests or "", "</test_cases_or_test_patch>",
        "", "<golden_patch>", golden_patch or "", "</golden_patch>",
        "", "Return ONLY a JSON object.",
    ])


def _build_keypoints_user_prompt(
    instance_id: str, resolved: bool, problem_statement: str,
    submission_patch: str, golden_patch: str, golden_tests: str,
    trajectory_text: str, candidate_api_paths: list[str], max_items: int,
) -> str:
    parts: list[str] = [
        "=== Instance ===",
        f"instance_id: {instance_id}", f"resolved: {str(resolved).lower()}", "",
        "=== Problem Statement ===", problem_statement or "", "",
        "=== Candidate Patch ===", submission_patch or "", "",
    ]
    if not resolved:
        parts += [
            "=== Golden Patch (unresolved only) ===", golden_patch or "", "",
            "=== Golden Tests (unresolved only) ===", golden_tests or "", "",
        ]
    parts += [
        "=== Agent Trajectory ===", trajectory_text or "", "",
        "=== Candidate API Paths (STRICT) ===",
        *[f"- {api}" for api in candidate_api_paths], "",
        "=== Task ===",
        "Return a JSON ARRAY only. Each item: api_path, experience_or_reflection, thinking.",
        "api_path MUST be from Candidate API Paths.",
        f"Output top {max_items} items.",
    ]
    return "\n".join(parts)


def _build_env_knowledge_user_prompt(
    instance_id: str, resolved: bool, problem_statement: str,
    submission_patch: str, golden_patch: str, golden_tests: str,
    trajectory_text: str, candidate_api_paths: list[str], max_items: int,
) -> str:
    parts: list[str] = [
        "=== Instance ===",
        f"instance_id: {instance_id}", f"resolved: {str(resolved).lower()}", "",
        "=== Problem Statement ===", problem_statement or "", "",
        "=== Candidate Patch ===", submission_patch or "", "",
    ]
    if not resolved:
        parts += [
            "=== Golden Patch (unresolved only) ===", golden_patch or "", "",
            "=== Golden Tests (unresolved only) ===", golden_tests or "", "",
        ]
    parts += [
        "=== Agent Trajectory ===", trajectory_text or "", "",
        "=== Candidate API Paths (STRICT) ===",
        *[f"- {api}" for api in candidate_api_paths], "",
        "=== Task ===",
        "Return a JSON ARRAY only. Each item: api_path, purpose, playbook, related_apis, thinking.",
        "api_path and related_apis MUST be from Candidate API Paths.",
        'related_apis: array of {"api_path": "...", "reason": "..."} objects.',
        f"Output top {max_items} items.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Issue-focus helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _clean_problem_statement(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^#+\s*.*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\*\*.*?\*\*\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^\s*(thanks|thank you).*$", " ", text, flags=re.MULTILINE)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return text[:1200].rstrip() if len(text) > 1200 else text


def _format_issue_focus_dict(data: dict[str, Any]) -> str | None:
    if not isinstance(data, dict):
        return None
    s = data.get("symptom", "").strip()
    e = data.get("expected_behavior", "").strip()
    c = data.get("constraints", "").strip()
    if not s:
        return None
    parts = [f"Symptom: {s}"]
    if e:
        parts.append(f"Expect: {e}")
    if c:
        parts.append(f"Constraints: {c}")
    return " ".join(" | ".join(parts).replace("\n", " ").replace("\r", " ").split())


def _normalize_issue_focus(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"```.*?```", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\*\*.*?\*\*\s*:?\s*$", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = " ".join(cleaned.replace("\r", " ").replace("\n", " ").split())
    return (cleaned[:240].rstrip() or None) if cleaned else None


def _looks_like_copied_problem_statement(summary: str, problem_statement: str) -> bool:
    if not summary or not problem_statement:
        return False
    a = {w.lower() for w in _WORD_RE.findall(summary)}
    b = {w.lower() for w in _WORD_RE.findall(problem_statement)}
    if not a or not b:
        return False
    return len(a & b) / max(1, len(a | b)) >= 0.75


# ---------------------------------------------------------------------------
# LLM call with structured logging
# ---------------------------------------------------------------------------


def _call_llm_with_logging(
    instance_id: str,
    resolved: bool,
    kind: str,
    messages: list[dict[str, str]],
    model: str,
    **llm_kwargs: Any,
) -> tuple[str, InstanceCallLog]:
    response = _query_with_retry(model, messages, **llm_kwargs)
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None) or {}
    log_entry = InstanceCallLog(
        instance_id=instance_id, resolved=resolved, kind=kind, model=model,
        messages=messages, output=content,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
    )
    return content, log_entry


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------


def _process_instance(
    instance_id: str,
    traj_file: str | Path,
    resolved: bool | None,
    leaderboard_path: str | Path,
    model: str,
    repo_root: str | None,
    extract_keypoints: bool,
    extract_env: bool,
    max_candidates: int,
    max_items: int,
    max_steps: int | None,
    llm_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[InstanceCallLog]]:
    traj = load_trajectory(traj_file)

    real_instance_id = instance_id
    original_instance_id = instance_id
    effective_repo_root = repo_root
    metadata_file = Path(traj_file).parent / f"{instance_id}_metadata.json"
    if metadata_file.exists():
        try:
            meta = json.loads(metadata_file.read_text())
            if resolved is None:
                verification = meta.get("verification")
                if isinstance(verification, dict):
                    resolved = verification.get("all_passed")
                if resolved is None:
                    resolved = meta.get("agent_success")
            original_instance_id = meta.get("original_instance_id", instance_id)
            bug_folder = meta.get("bug_folder", "")
            if bug_folder:
                real_instance_id = Path(bug_folder).name
                if not effective_repo_root:
                    potential_repo = Path(bug_folder) / "repo"
                    effective_repo_root = (
                        str(potential_repo) if potential_repo.exists() else bug_folder
                    )
        except Exception as exc:
            print(f"Warning: failed to parse metadata for {instance_id}: {exc}")

    golden = _load_golden_info(Path(leaderboard_path), real_instance_id)
    submission_patch = (traj.info or {}).get("submission", "") or ""

    if resolved is None:
        resolved = _check_if_resolved_by_llm(
            real_instance_id, golden.problem_statement, golden.patch,
            submission_patch, model, **llm_kwargs,
        )

    assert isinstance(resolved, bool), f"resolved must be bool, got {type(resolved)}"

    issue_text = golden.problem_statement or ""
    trajectory_text = _build_trajectory_text(traj, max_steps=max_steps)

    candidate_api_paths = _build_candidate_api_paths(
        traj=traj, golden=golden, submission_patch=submission_patch,
        repo_root=Path(effective_repo_root) if effective_repo_root else None,
        max_candidates=max_candidates,
    )

    if effective_repo_root:
        rr_path = Path(effective_repo_root).resolve()
        cleaned_candidates: list[str] = []
        for cp in candidate_api_paths:
            filepath_part = cp.split("::")[0]
            try:
                p = Path(filepath_part).resolve()
                if p.is_relative_to(rr_path):
                    filepath_part = str(p.relative_to(rr_path))
                elif filepath_part.startswith("repo/"):
                    filepath_part = filepath_part[5:]
            except Exception:
                if filepath_part.startswith("repo/"):
                    filepath_part = filepath_part[5:]
            cleaned_candidates.append(cp.replace(cp.split("::")[0], filepath_part, 1))
        candidate_api_paths = cleaned_candidates

    candidate_set = set(candidate_api_paths)
    records: list[dict[str, Any]] = []
    logs: list[InstanceCallLog] = []
    max_attempts = 3

    # Issue-focus normalisation
    if issue_text:
        ps_clean = _clean_problem_statement(golden.problem_statement or "")
        msgs = [
            {"role": "system", "content": _ISSUE_FOCUS_SYSTEM},
            {"role": "user", "content": _build_issue_focus_user_prompt(
                problem_statement=golden.problem_statement,
                golden_patch=golden.patch, golden_tests=golden.test_patch,
            )},
        ]
        issue_focus: str | None = None
        for _ in range(max_attempts):
            content, log = _call_llm_with_logging(
                real_instance_id, resolved, "issue_focus", msgs, model, **llm_kwargs
            )
            logs.append(log)
            obj = _parse_json_dict(content)
            if obj:
                formatted = _format_issue_focus_dict(obj)
                if formatted and not _looks_like_copied_problem_statement(formatted, ps_clean):
                    issue_focus = formatted
                    break
        issue_text = issue_focus or _normalize_issue_focus(issue_text) or issue_text

    # Local intervention skill extraction
    if extract_keypoints:
        msgs = [
            {"role": "system", "content": _KP_SYSTEM},
            {"role": "user", "content": _build_keypoints_user_prompt(
                instance_id=real_instance_id, resolved=resolved,
                problem_statement=golden.problem_statement,
                submission_patch=submission_patch, golden_patch=golden.patch,
                golden_tests=golden.test_patch, trajectory_text=trajectory_text,
                candidate_api_paths=candidate_api_paths, max_items=max_items,
            )},
        ]
        parsed: list[dict[str, Any]] | None = None
        for _ in range(max_attempts):
            content, log = _call_llm_with_logging(
                real_instance_id, resolved, "keypoints", msgs, model, **llm_kwargs
            )
            logs.append(log)
            candidate = _parse_json_array(content)
            if candidate is not None:
                validated = _validate_keypoints(candidate, candidate_set)
                if validated is not None:
                    parsed = [{**it, "issue": issue_text, "original_instance_id": original_instance_id} for it in validated]
                    break
        records.append({
            "instance_id": real_instance_id, "original_instance_id": original_instance_id,
            "kind": "keypoints", "resolved": resolved,
            "candidate_api_paths": candidate_api_paths, "items": parsed or [],
        })

    # Env-knowledge extraction
    if extract_env:
        msgs = [
            {"role": "system", "content": _ENV_SYSTEM},
            {"role": "user", "content": _build_env_knowledge_user_prompt(
                instance_id=real_instance_id, resolved=resolved,
                problem_statement=golden.problem_statement,
                submission_patch=submission_patch, golden_patch=golden.patch,
                golden_tests=golden.test_patch, trajectory_text=trajectory_text,
                candidate_api_paths=candidate_api_paths, max_items=max_items,
            )},
        ]
        parsed = None
        for _ in range(max_attempts):
            content, log = _call_llm_with_logging(
                real_instance_id, resolved, "env_knowledge", msgs, model, **llm_kwargs
            )
            logs.append(log)
            candidate = _parse_json_array(content)
            if candidate is not None:
                validated = _validate_env_knowledge(candidate, candidate_set)
                if validated is not None:
                    parsed = [{**it, "original_instance_id": original_instance_id} for it in validated]
                    break
        records.append({
            "instance_id": real_instance_id, "original_instance_id": original_instance_id,
            "kind": "env_knowledge", "resolved": resolved,
            "candidate_api_paths": candidate_api_paths, "items": parsed or [],
        })

    return records, logs


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_experiences(
    traj_root: str | Path,
    leaderboard_path: str | Path,
    output_dir: str | Path,
    log_path: str | Path,
    model: str,
    eval_json_path: str | Path | None = None,
    repo_root: str | None = None,
    extract_keypoints: bool = True,
    extract_env: bool = False,
    max_instances: int | None = None,
    max_workers: int = 4,
    max_candidates: int = 200,
    max_items: int = 10,
    max_steps: int | None = None,
    **llm_kwargs: Any,
) -> None:
    """
    Orchestrate parallel per-instance extraction and write results to output_dir.

    Output files
    ------------
    keypoints.jsonl      one line per instance  (if extract_keypoints=True)
    env_knowledge.jsonl  one line per instance  (if extract_env=True)
    <log_path>           Markdown log of every LLM call
    """
    out_dir = Path(output_dir)
    log_file = Path(log_path)

    resolved_ids, unresolved_ids = _load_eval_ids(Path(eval_json_path) if eval_json_path else None)
    traj_files = _iter_traj_instances(Path(traj_root))

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    items = sorted(traj_files.items())
    if max_instances is not None:
        items = items[:max_instances]

    records: list[dict[str, Any]] = []
    logs: list[InstanceCallLog] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id: dict[Any, str] = {}
        for instance_id, traj_file in items:
            resolved: bool | None = None
            if instance_id in resolved_ids:
                resolved = True
            elif instance_id in unresolved_ids:
                resolved = False
            future = executor.submit(
                _process_instance,
                instance_id, str(traj_file), resolved, str(leaderboard_path),
                model, repo_root, extract_keypoints, extract_env,
                max_candidates, max_items, max_steps, llm_kwargs,
            )
            future_to_id[future] = instance_id

        for future in as_completed(future_to_id):
            try:
                recs, call_logs = future.result()
                records.extend(recs)
                logs.extend(call_logs)
            except Exception as exc:
                print(f"Error processing {future_to_id[future]}: {exc}")

    if extract_keypoints:
        with (out_dir / "keypoints.jsonl").open("a", encoding="utf-8") as fh:
            for rec in (r for r in records if r["kind"] == "keypoints"):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if extract_env:
        with (out_dir / "env_knowledge.jsonl").open("a", encoding="utf-8") as fh:
            for rec in (r for r in records if r["kind"] == "env_knowledge"):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with log_file.open("a", encoding="utf-8") as fh:
        for entry in sorted(logs, key=lambda e: (e.instance_id, e.kind)):
            fh.write(entry.to_markdown())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-instance keypoints / env-knowledge from agent trajectories."
    )
    parser.add_argument("--traj-root", required=True)
    parser.add_argument("--leaderboard", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument(
        "--eval-json", default=None,
        help="Optional pre-computed eval JSON with resolved_ids / unresolved_ids.",
    )
    parser.add_argument(
        "--api-base", default=None,
        help="LLM API base URL (or set OPENAI_BASE_URL env var).",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="LLM API key (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--extract-keypoints", "--extract-local-skills", dest="extract_keypoints", action="store_true")
    parser.add_argument("--extract-env", "--extract-global-skills", dest="extract_env", action="store_true")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--max-workers", "-w", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=200)
    parser.add_argument("--max-items", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)

    args = parser.parse_args()
    extract_keypoints = bool(args.extract_keypoints or not args.extract_env)

    llm_kwargs: dict[str, Any] = {}
    if args.api_base:
        llm_kwargs["api_base"] = args.api_base
    if args.api_key:
        llm_kwargs["api_key"] = args.api_key

    extract_experiences(
        traj_root=args.traj_root, leaderboard_path=args.leaderboard,
        output_dir=args.output_dir, log_path=args.log, model=args.model,
        eval_json_path=args.eval_json, repo_root=args.repo_root,
        extract_keypoints=extract_keypoints, extract_env=args.extract_env,
        max_instances=args.max_instances, max_workers=args.max_workers,
        max_candidates=args.max_candidates, max_items=args.max_items,
        max_steps=args.max_steps, **llm_kwargs,
    )


if __name__ == "__main__":
    main()
