#!/usr/bin/env python3
"""
Extract existing tests from SWE-bench instances.

For each instance already set up by setup_repos.py, inspects FAIL_TO_PASS and
test_patch to find test functions that exist in the base_commit (not newly added
by the patch), and writes a JSON mapping of instance_id -> [pytest_paths].

This output is used as input to verify_tests.py and generate_bugs.py.

Usage:
    python synthesis/extract_tests.py \\
        --work-dir synthesis/workdir \\
        --output synthesis/workdir/target_tests.json
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_pytest_path(test_str: str) -> tuple[str, str, str]:
    """
    Parse a pytest test path into (file_path, class_name, test_name).
    Handles both 'file.py::test_fn' and 'file.py::Class::test_fn'.
    """
    if "::" not in test_str:
        return "", "", test_str
    parts = test_str.split("::")
    file_path = parts[0]
    if len(parts) == 2:
        return file_path, "", parts[1]
    return file_path, parts[1], parts[2]


def extract_test_file_from_patch(test_patch: str) -> str:
    if not test_patch:
        return ""
    match = re.search(r"diff --git a/(.*?\.py)", test_patch)
    return match.group(1) if match else ""


def extract_test_functions_from_patch(test_patch: str) -> set[str]:
    functions: set[str] = set()
    if not test_patch:
        return functions
    for line in test_patch.splitlines():
        m = re.match(r"^\+?\s*def\s+(test_\w+)\s*\(", line)
        if m:
            functions.add(m.group(1))
        if line.startswith("@@"):
            functions.update(re.findall(r"\b(test_\w+)", line))
    return functions


def function_exists_in_file(file_path: Path, test_name: str, class_name: str = "") -> bool:
    if not file_path.exists():
        return False
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False
    if class_name:
        if not re.search(rf"^\s*class\s+{re.escape(class_name)}\s*[:\(]", content, re.MULTILINE):
            return False
        return bool(re.search(rf"^\s+def\s+{re.escape(test_name)}\s*\(", content, re.MULTILINE))
    return bool(re.search(rf"^\s*def\s+{re.escape(test_name)}\s*\(", content, re.MULTILINE))


def extract_existing_tests_for_instance(instance_id: str, repos_dir: Path) -> list[str]:
    instance_dir = repos_dir / instance_id
    data_file = instance_dir / "instance_data.json"
    repo_path = instance_dir / "repo"

    if not data_file.exists() or not repo_path.exists():
        logger.warning(f"{instance_id}: repo or instance_data.json not found, skipping")
        return []

    instance_data = json.loads(data_file.read_text())
    test_patch = instance_data.get("test_patch", "")
    patch_file = extract_test_file_from_patch(test_patch)
    patch_tests = extract_test_functions_from_patch(test_patch)

    # Candidates from FAIL_TO_PASS (has file + class + method)
    ftp_raw = instance_data.get("FAIL_TO_PASS", "[]")
    fail_to_pass = json.loads(ftp_raw) if isinstance(ftp_raw, str) else ftp_raw
    candidates: list[tuple[str, str, str]] = []
    for test_str in fail_to_pass or []:
        fp, cls, fn = parse_pytest_path(test_str)
        candidates.append((fp, cls, fn))
    # Supplement from patch (no class info)
    for fn in patch_tests:
        candidates.append((patch_file, "", fn))

    existing: list[str] = []
    seen: set[str] = set()
    for file_rel, class_name, test_name in candidates:
        if not test_name:
            continue
        file_use = file_rel or patch_file
        if not file_use:
            continue
        if not function_exists_in_file(repo_path / file_use, test_name, class_name):
            continue
        node = f"{class_name}::{test_name}" if class_name else test_name
        pytest_path = f"{file_use}::{node}"
        if pytest_path not in seen:
            seen.add(pytest_path)
            existing.append(pytest_path)
            logger.info(f"  ✓ {pytest_path}")

    return existing


def main():
    parser = argparse.ArgumentParser(description="Extract existing tests from SWE-bench instances")
    parser.add_argument("--work-dir", type=Path, default=Path("synthesis/workdir"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON (defaults to work-dir/target_tests.json)",
    )
    parser.add_argument(
        "--filter",
        default="",
        dest="filter_spec",
        help="Only process instances whose ID contains this substring",
    )
    args = parser.parse_args()

    repos_dir = args.work_dir / "repos"
    output_file = args.output or (args.work_dir / "target_tests.json")

    if not repos_dir.exists():
        logger.error(f"repos dir not found: {repos_dir}. Run setup_repos.py first.")
        return

    instance_ids = sorted(d.name for d in repos_dir.iterdir() if d.is_dir())
    if args.filter_spec:
        instance_ids = [i for i in instance_ids if args.filter_spec in i]
    logger.info(f"Processing {len(instance_ids)} instances")

    results: dict[str, list[str]] = {}
    for i, instance_id in enumerate(instance_ids, 1):
        logger.info(f"\n[{i}/{len(instance_ids)}] {instance_id}")
        tests = extract_existing_tests_for_instance(instance_id, repos_dir)
        if tests:
            results[instance_id] = tests

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))

    total = sum(len(v) for v in results.values())
    logger.info(f"\nDone: {len(results)}/{len(instance_ids)} instances have tests, {total} tests total")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()
