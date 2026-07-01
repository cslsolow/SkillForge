"""
Extract code-access events (file path + optional line range) from each step
of an agent trajectory by parsing the bash actions and their outputs.
"""

from __future__ import annotations

import re

from .schema import AccessType, CodeAccess
from .source_files import SOURCE_EXTENSIONS, is_source_file

_ROOT_PREFIXES = ("/testbed/", "./", "/testbed")


def _normalize_filepath(path: str) -> str:
    for prefix in _ROOT_PREFIXES:
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    return path.lstrip("/")


_SOURCE_EXT_PATTERN = "|".join(re.escape(ext) for ext in SOURCE_EXTENSIONS)
_SOURCE_PATH_PATTERN = rf"(?:(?:/|\./)?[\w.\-@]+/)*[\w.\-@]+(?:{_SOURCE_EXT_PATTERN})"


def _extract_filepaths_from_find(output: str) -> list[str]:
    return [
        _normalize_filepath(line.strip())
        for line in output.strip().splitlines()
        if is_source_file(line.strip())
    ]


def _parse_grep_command(action: str) -> tuple[str | None, str | None]:
    match = re.search(r'grep\s+(?:-\w+\s+)*["\']?([^"\']+)["\']?\s+(\S+)', action)
    if match:
        return _normalize_filepath(match.group(2)), match.group(1)
    match = re.search(r"(\S+)\s*$", action)
    if match:
        target = match.group(1)
        if is_source_file(target):
            return _normalize_filepath(target), None
    return None, None


def _parse_grep_output(output: str) -> list[tuple[str, int]]:
    results: list[tuple[str, int]] = []
    for line in output.strip().splitlines():
        m = re.match(rf"^(\S+(?:{_SOURCE_EXT_PATTERN})):(\d+):", line)
        if m:
            results.append((_normalize_filepath(m.group(1)), int(m.group(2))))
            continue
        m = re.match(r"^(\d+):", line)
        if m:
            results.append(("", int(m.group(1))))
    return results


def _parse_sed_view_command(action: str) -> tuple[str | None, int | None, int | None]:
    m = re.search(r"sed\s+-n\s+['\"]?(\d+),(\d+)p['\"]?\s+(\S+)", action)
    if m:
        return _normalize_filepath(m.group(3)), int(m.group(1)), int(m.group(2))
    m = re.search(r"sed\s+-n\s+['\"]?(\d+)p['\"]?\s+(\S+)", action)
    if m:
        return _normalize_filepath(m.group(2)), int(m.group(1)), int(m.group(1))
    return None, None, None


def _parse_nl_sed_command(action: str) -> tuple[str | None, int | None, int | None]:
    m = re.search(
        rf"nl\s+(?:-\w+\s+)*({_SOURCE_PATH_PATTERN})\s*\|\s*sed\s+-n\s+['\"]?(\d+),(\d+)p", action
    )
    if m:
        return _normalize_filepath(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(
        rf"nl\s+(?:-\w+\s+)*({_SOURCE_PATH_PATTERN})\s*\|\s*sed\s+-n\s+['\"]?(\d+)p", action
    )
    if m:
        return _normalize_filepath(m.group(1)), int(m.group(2)), int(m.group(2))
    return None, None, None


def _parse_cat_command(action: str) -> str | None:
    m = re.search(rf"\bcat\s+({_SOURCE_PATH_PATTERN})", action)
    return _normalize_filepath(m.group(1)) if m else None


def _parse_head_tail_command(action: str) -> tuple[str | None, int | None, int | None]:
    m = re.search(rf"\bhead\s+-n\s+(\d+)\s+({_SOURCE_PATH_PATTERN})", action)
    if m:
        return _normalize_filepath(m.group(2)), 1, int(m.group(1))
    m = re.search(rf"\btail\s+-n\s+(\d+)\s+({_SOURCE_PATH_PATTERN})", action)
    if m:
        return _normalize_filepath(m.group(2)), None, None
    return None, None, None


def _parse_sed_edit_command(action: str) -> tuple[str | None, int | None, int | None]:
    m = re.search(rf"sed\s+-i\s+.*\s+({_SOURCE_PATH_PATTERN})", action)
    if not m:
        return None, None, None
    filepath = _normalize_filepath(m.group(1))
    lm = re.search(r"sed\s+-i\s+['\"]?(\d+),(\d+)", action)
    if lm:
        return filepath, int(lm.group(1)), int(lm.group(2))
    lm = re.search(r"sed\s+-i\s+['\"]?(\d+)", action)
    if lm:
        return filepath, int(lm.group(1)), int(lm.group(1))
    return filepath, None, None


def extract_code_accesses_from_step(
    step_index: int,
    thought: str,
    action: str,
    observation: str,
) -> list[CodeAccess]:
    accesses: list[CodeAccess] = []

    if "find " in action and any(ext in observation for ext in SOURCE_EXTENSIONS):
        for fp in _extract_filepaths_from_find(observation):
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.SEARCH, thought=thought, action=action))

    if "grep " in action:
        fp, _ = _parse_grep_command(action)
        grep_results = _parse_grep_output(observation)
        if grep_results:
            for result_fp, lineno in grep_results:
                final_fp = result_fp or fp
                if final_fp:
                    accesses.append(CodeAccess(step_index=step_index, filepath=final_fp, access_type=AccessType.SEARCH, start_line=lineno, end_line=lineno, thought=thought, action=action))
        elif fp:
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.SEARCH, thought=thought, action=action))

    if "sed -n" in action or "sed  -n" in action:
        fp, start, end = _parse_sed_view_command(action)
        if fp:
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.VIEW, start_line=start, end_line=end, thought=thought, action=action))

    if "nl " in action and "sed -n" in action:
        fp, start, end = _parse_nl_sed_command(action)
        if fp:
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.VIEW, start_line=start, end_line=end, thought=thought, action=action))

    if "cat " in action and "sed -n" not in action:
        fp = _parse_cat_command(action)
        if fp:
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.VIEW, thought=thought, action=action))

    if "head " in action or "tail " in action:
        fp, start, end = _parse_head_tail_command(action)
        if fp:
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.VIEW, start_line=start, end_line=end, thought=thought, action=action))

    if "sed -i" in action or "sed  -i" in action:
        fp, start, end = _parse_sed_edit_command(action)
        if fp:
            accesses.append(CodeAccess(step_index=step_index, filepath=fp, access_type=AccessType.EDIT, start_line=start, end_line=end, thought=thought, action=action))

    return accesses


def extract_all_code_accesses(steps: list) -> list[CodeAccess]:
    all_accesses: list[CodeAccess] = []
    for step in steps:
        all_accesses.extend(
            extract_code_accesses_from_step(
                step_index=step.index,
                thought=step.thought,
                action=step.action,
                observation=step.observation,
            )
        )
    return all_accesses
