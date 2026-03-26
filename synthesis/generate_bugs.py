#!/usr/bin/env python3
"""
Generate buggy instances by invoking strict_mask_generator.py for each verified test.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL) to be set in the environment.

Usage:
    python synthesis/generate_bugs.py \\
        --target-tests synthesis/workdir/target_tests_verified.json \\
        --work-dir synthesis/workdir \\
        --generator synthesis/strict_mask_generator.py \\
        --timeout 3000
"""
import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_GENERATOR = Path(__file__).parent / "strict_mask_generator.py"


def generate_for_test(
    instance_id: str,
    test_str: str,
    repos_dir: Path,
    output_base: Path,
    generator_script: Path,
    timeout: int,
) -> bool:
    repo_dir = repos_dir / instance_id / "repo"
    python_path = repos_dir / instance_id / "venv" / "bin" / "python"

    if not repo_dir.exists():
        logger.error(f"Repo not found: {repo_dir}")
        return False
    if not python_path.exists():
        logger.error(f"Python not found: {python_path}")
        return False

    parts = test_str.split("::")
    if len(parts) < 2:
        logger.error(f"Cannot parse test string: {test_str}")
        return False

    file_path = parts[0]
    module_name = file_path.removesuffix(".py").replace("/", ".")
    class_name = parts[1] if len(parts) == 3 else ""
    method_name = parts[-1]

    safe_test_id = test_str.replace("::", "__").replace("/", "_").replace(".", "_")
    test_output_dir = output_base / instance_id / safe_test_id
    cmd = [
        str(python_path),
        str(generator_script),
        "--repo", str(repo_dir),
        "--test-module", module_name,
        "--test-class", class_name,
        "--test-method", method_name,
        "--output", str(test_output_dir),
    ]

    env = os.environ.copy()
    env["PYTEST_ADDOPTS"] = "-W ignore::DeprecationWarning -W ignore::FutureWarning"

    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout, env=env)
        if result.returncode == 0:
            logger.info(f"  ✓ {test_str}")
            return True
        else:
            logger.error(f"  ✗ {test_str} (rc={result.returncode})")
            if result.stderr:
                logger.debug(result.stderr[:300])
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"  ⏱ {test_str} (timeout >{timeout}s)")
        return False
    except Exception as e:
        logger.error(f"  ⚠ {test_str}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate buggy instances from verified tests")
    parser.add_argument("--target-tests", type=Path, required=True, help="JSON: instance_id -> [test_paths] (verified)")
    parser.add_argument("--work-dir", type=Path, default=Path("synthesis/workdir"))
    parser.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR, help="Path to strict_mask_generator.py")
    parser.add_argument("--timeout", type=int, default=3000, help="Timeout per test in seconds")
    args = parser.parse_args()

    if not args.generator.exists():
        logger.error(f"Generator not found: {args.generator}")
        return

    repos_dir = args.work_dir / "repos"
    output_base = args.work_dir / "bugs_from_patch"
    output_base.mkdir(parents=True, exist_ok=True)

    with open(args.target_tests) as f:
        target_tests_map = json.load(f)

    total = success = failed = 0
    for instance_id, tests in sorted(target_tests_map.items()):
        logger.info(f"\n{'='*60}")
        logger.info(f"{instance_id}  ({len(tests)} tests)")
        for test_str in tests:
            ok = generate_for_test(instance_id, test_str, repos_dir, output_base, args.generator, args.timeout)
            total += 1
            if ok:
                success += 1
            else:
                failed += 1

    logger.info(f"\nBug generation complete: {success}/{total} succeeded")
    logger.info(f"Output: {output_base}")


if __name__ == "__main__":
    main()
