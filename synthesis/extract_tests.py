#!/usr/bin/env python3
"""
Extract all test cases from a repository for bug generation.

Supports two modes:
1. User-specified tests: Provide a JSON file mapping repo_name -> [test_paths]
2. Repo-scan mode: Discover all tests via pytest --collect-only from cloned repos

Usage:
    # User-specified tests
    python synthesis/extract_tests.py \
        --user-tests tests_config.json \
        --output synthesis/workdir/target_tests.json

    # Repo-scan mode (discover all tests from repos)
    python synthesis/extract_tests.py \
        --work-dir synthesis/workdir \
        --output synthesis/workdir/target_tests.json
"""
import argparse
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def discover_all_tests_in_repo(
    instance_id: str,
    repos_dir: Path,
    timeout: int = 120,
) -> list[str]:
    """Discover all test cases in a repository using pytest --collect-only.

    Returns a list of pytest node IDs like:
        tests/test_foo.py::TestClass::test_method
        tests/test_bar.py::test_function
    """
    repo_path = repos_dir / instance_id / "repo"
    venv_python = repos_dir / instance_id / "venv" / "bin" / "python"

    if not repo_path.exists():
        logger.warning(f"{instance_id}: repo not found at {repo_path}, skipping")
        return []
    if not venv_python.exists():
        logger.warning(f"{instance_id}: venv python not found at {venv_python}, skipping")
        return []

    cmd = [
        str(venv_python), "-m", "pytest",
        "--collect-only", "-q",
        "--rootdir", str(repo_path),
        "--ignore=venv",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"{instance_id}: pytest --collect-only timed out after {timeout}s")
        return []
    except Exception as exc:
        logger.error(f"{instance_id}: failed to run pytest --collect-only: {exc}")
        return []

    tests: list[str] = []
    seen: set[str] = set()

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(("=", "-", "no tests", "<")):
            continue
        if "::" not in line:
            continue
        # Remove trailing parametrize markers for deduplication, but keep full node id
        node_id = line.split("[")[0] if "[" in line else line
        if node_id in seen:
            continue
        # Basic sanity: must contain a .py file reference
        if not re.match(r".*\.py::", node_id):
            continue
        seen.add(node_id)
        tests.append(node_id)

    logger.info(f"  Discovered {len(tests)} unique test cases in {instance_id}")
    return tests


def main():
    parser = argparse.ArgumentParser(description="Extract all test cases from repositories for bug generation")
    parser.add_argument(
        "--user-tests",
        type=Path,
        default=None,
        help='JSON file with user-specified tests: {"repo_name": ["path/to/test.py::TestClass::test_method"]}',
    )
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
        help="Only process instances whose ID contains this substring (repo-scan mode only)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for test collection per instance",
    )
    parser.add_argument(
        "--language",
        default="python",
        help="Repository language: python (default), go, typescript",
    )
    args = parser.parse_args()

    output_file = args.output or (args.work_dir / "target_tests.json")

    # Mode 1: User-specified tests
    if args.user_tests:
        if not args.user_tests.exists():
            logger.error(f"User tests file not found: {args.user_tests}")
            return

        logger.info(f"Loading user-specified tests from {args.user_tests}")
        results = json.loads(args.user_tests.read_text())
        logger.info(f"Loaded {len(results)} test configurations")

    # Mode 2: Repo-scan mode — discover tests for the given language
    else:
        repos_dir = args.work_dir / "repos"
        if not repos_dir.exists():
            logger.error(f"repos dir not found: {repos_dir}. Run setup_repos.py first or use --user-tests.")
            return

        instance_ids = sorted(d.name for d in repos_dir.iterdir() if d.is_dir())
        if args.filter_spec:
            instance_ids = [i for i in instance_ids if args.filter_spec in i]
        logger.info(f"Scanning {len(instance_ids)} repositories for test cases (language={args.language})")

        if args.language == "python":
            discover_fn = lambda iid: discover_all_tests_in_repo(iid, repos_dir, timeout=args.timeout)
        else:
            from language_adapters import get_adapter
            _adapter = get_adapter(args.language)
            discover_fn = lambda iid: _adapter.discover_tests(iid, repos_dir, timeout=args.timeout)

        results: dict[str, list[str]] = {}
        for i, instance_id in enumerate(instance_ids, 1):
            logger.info(f"\n[{i}/{len(instance_ids)}] {instance_id}")
            tests = discover_fn(instance_id)
            logger.info(f"  Discovered {len(tests)} unique test cases in {instance_id}")
            if tests:
                results[instance_id] = tests

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))

    total = sum(len(v) for v in results.values())
    logger.info(f"\nDone: {len(results)} configurations, {total} tests total")
    logger.info(f"Output: {output_file}")


if __name__ == "__main__":
    main()
