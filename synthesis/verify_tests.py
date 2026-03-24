#!/usr/bin/env python3
"""
Verify that extracted test cases pass on the base_commit for each instance.

Usage:
    python synthesis/verify_tests.py \\
        --target-tests synthesis/workdir/target_tests.json \\
        --work-dir synthesis/workdir \\
        --output synthesis/workdir/test_verification_results.json
"""
import argparse
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_tests_for_instance(instance_id: str, tests: list, repos_dir: Path, timeout: int = 120) -> dict:
    venv_python = repos_dir / instance_id / "venv" / "bin" / "python"
    repo_path = repos_dir / instance_id / "repo"

    if not venv_python.exists():
        return {
            "instance_id": instance_id,
            "total_tests": len(tests),
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "error": "Virtual environment not found",
            "test_results": {},
            "verified_tests": []
        }

    test_results = {}
    verified_tests = []
    passed = failed = errors = 0

    for test_path in tests:
        cmd = [str(venv_python), "-m", "pytest", "-xvs", test_path, "--tb=short", "--no-header"]
        try:
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                test_results[test_path] = {"status": "PASSED", "output": ""}
                verified_tests.append(test_path)
                passed += 1
                logger.info(f"  ✓ {test_path}")
            else:
                test_results[test_path] = {"status": "FAILED", "output": result.stdout + result.stderr}
                failed += 1
                logger.warning(f"  ✗ {test_path}")
        except subprocess.TimeoutExpired:
            test_results[test_path] = {"status": "TIMEOUT", "output": f"Test timed out after {timeout}s"}
            errors += 1
            logger.error(f"  ⏱ {test_path}")
        except Exception as e:
            test_results[test_path] = {"status": "ERROR", "output": str(e)}
            errors += 1
            logger.error(f"  ⚠ {test_path}: {e}")

    return {
        "instance_id": instance_id,
        "total_tests": len(tests),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "test_results": test_results,
        "verified_tests": verified_tests
    }


def main():
    parser = argparse.ArgumentParser(description="Verify extracted tests on base_commit")
    parser.add_argument("--target-tests", type=Path, required=True, help="JSON file mapping instance_id -> [test_paths]")
    parser.add_argument("--work-dir", type=Path, default=Path("synthesis/workdir"))
    parser.add_argument("--output", type=Path, default=None, help="Output JSON file (defaults to work-dir/test_verification_results.json)")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per test in seconds")
    args = parser.parse_args()

    output_file = args.output or (args.work_dir / "test_verification_results.json")
    repos_dir = args.work_dir / "repos"

    with open(args.target_tests) as f:
        target_tests = json.load(f)

    logger.info(f"Testing {len(target_tests)} instances")

    results = {}
    total_passed = total_failed = total_errors = total_tests = 0

    for i, (instance_id, tests) in enumerate(sorted(target_tests.items()), 1):
        logger.info(f"\n[{i}/{len(target_tests)}] {instance_id} ({len(tests)} tests)")
        result = run_tests_for_instance(instance_id, tests, repos_dir, args.timeout)
        results[instance_id] = result
        total_tests += result["total_tests"]
        total_passed += result["passed"]
        total_failed += result["failed"]
        total_errors += result["errors"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, indent=2))

    # Create a simplified mapping for generate_bugs.py: instance_id -> [verified_test_paths]
    verified_mapping = {}
    for instance_id, result in results.items():
        verified_tests = result.get("verified_tests", [])
        if verified_tests:
            verified_mapping[instance_id] = verified_tests

    verified_output = args.work_dir / "target_tests_verified.json"
    verified_output.write_text(json.dumps(verified_mapping, indent=2))

    logger.info(f"\nVerification complete: {total_passed}/{total_tests} passed")
    logger.info(f"Detailed results saved to {output_file}")
    logger.info(f"Verified tests mapping saved to {verified_output}")


if __name__ == "__main__":
    main()
