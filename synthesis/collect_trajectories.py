#!/usr/bin/env python3
"""
Run mini-swe-agent on each bug instance and collect repair trajectories.

Requires OPENAI_API_KEY (and optionally OPENAI_BASE_URL, MSWEA_MODEL_NAME) in the environment.

Usage:
    python synthesis/collect_trajectories.py \\
        --instances synthesis/workdir/instances_for_trajectory.jsonl \\
        --work-dir synthesis/workdir \\
        --config src/minisweagent/config/extra/swebench_exp.yaml \\
        --model gpt-5-mini \\
        --timeout 7200
"""
import argparse
import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cleanup_repo(repo_path: Path):
    subprocess.run(["git", "reset", "--hard"], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "clean", "-fdx", "-e", "venv"], cwd=repo_path, capture_output=True)


def prepare_repo(repo_path: Path, base_commit: str, buggy_patch: str) -> bool:
    try:
        cleanup_repo(repo_path)
        subprocess.run(["git", "checkout", base_commit], cwd=repo_path, check=True, capture_output=True)

        patch_file = repo_path / ".buggy.patch"
        patch_file.write_text(buggy_patch)
        result = subprocess.run(
            ["git", "apply", str(patch_file)], cwd=repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["git", "apply", "--reject", "--whitespace=fix", str(patch_file)],
                cwd=repo_path, capture_output=True, text=True,
            )
            if result.returncode != 0:
                logger.error(f"Failed to apply patch: {result.stderr[:300]}")
                return False

        subprocess.run(["git", "add", "-A"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Apply buggy patch"], cwd=repo_path, check=True, capture_output=True)
        return True
    except Exception as e:
        logger.error(f"prepare_repo error: {e}")
        return False


def run_agent(
    repo_path: Path,
    venv_path: Path,
    instance_id: str,
    problem_statement: str,
    output_file: Path,
    log_file: Path,
    config_path: Path,
    model: str,
    timeout: int,
) -> bool:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path.resolve())
    env["PATH"] = f"{venv_path.resolve()}/bin:{env['PATH']}"
    env["PYTHONUNBUFFERED"] = "1"
    if model:
        env["MSWEA_MODEL_NAME"] = model

    cmd = ["mini", "-c", str(config_path), "-t", problem_statement, "-o", str(output_file), "-y"]
    logger.info(f"Running agent for {instance_id}  model={env.get('MSWEA_MODEL_NAME', 'default')}")

    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, cwd=repo_path, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        try:
            proc.wait(timeout=timeout)
            return proc.returncode == 0
        except subprocess.TimeoutExpired:
            proc.kill()
            logger.error(f"Agent timed out after {timeout}s")
            return False


def verify_fix(repo_path: Path, venv_path: Path, fail_to_pass: list) -> dict:
    if not fail_to_pass:
        return {"verified": False, "reason": "no_tests"}

    test_spec = fail_to_pass[0]
    if "::::" in test_spec:
        module_path, test_name = test_spec.split("::::")
        file_path = module_path.replace(".", "/") + ".py"
        test_command = f"{file_path}::{test_name}"
    else:
        test_command = test_spec

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_path.resolve())
    env["PATH"] = f"{venv_path.resolve()}/bin:{env['PATH']}"
    env["PYTHONPATH"] = str(repo_path.resolve())

    cmd = [str(venv_path / "bin" / "python"), "-m", "pytest", test_command, "-xvs"]
    try:
        result = subprocess.run(cmd, cwd=repo_path, env=env, capture_output=True, text=True, timeout=300)
        passed = result.returncode == 0
        return {"verified": True, "all_passed": passed, "test": test_command}
    except Exception as e:
        return {"verified": False, "reason": str(e)}


def process_instance(instance: dict, repos_dir: Path, output_dir: Path, config_path: Path, model: str, timeout: int) -> bool:
    instance_id = instance["instance_id"]
    original_id = instance.get("original_instance_id", instance_id)

    repo_path = repos_dir / original_id / "repo"
    venv_path = repos_dir / original_id / "venv"
    if not repo_path.exists():
        logger.error(f"Repo not found: {repo_path}")
        return False

    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    traj_file = instance_dir / f"{instance_id}_trajectory.json"
    log_file = instance_dir / f"{instance_id}_agent.log"
    meta_file = instance_dir / f"{instance_id}_metadata.json"

    if traj_file.exists():
        logger.info(f"Skipping {instance_id} (trajectory exists)")
        return True

    buggy_patch = instance.get("buggy_patch") or instance.get("patch", "")
    if not buggy_patch:
        logger.error(f"No buggy_patch for {instance_id}")
        return False

    if not prepare_repo(repo_path, instance["base_commit"], buggy_patch):
        return False

    problem_statement = instance.get("problem_statement") or instance.get("issue_statement", "")
    success = run_agent(repo_path, venv_path, instance_id, problem_statement, traj_file, log_file, config_path, model, timeout)

    git_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_path, capture_output=True, text=True
    ).stdout.strip()

    patch = ""
    verification: dict = {"verified": False, "reason": "no_changes"}
    if git_status:
        subprocess.run(["git", "add", "-A"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Agent fix"], cwd=repo_path, capture_output=True)
        diff = subprocess.run(["git", "diff", "HEAD^", "HEAD"], cwd=repo_path, capture_output=True, text=True)
        patch = diff.stdout

        ftp_raw = instance.get("FAIL_TO_PASS", "[]")
        ftp = json.loads(ftp_raw) if isinstance(ftp_raw, str) else ftp_raw
        try:
            verification = verify_fix(repo_path, venv_path, ftp)
        except Exception as e:
            verification = {"verified": False, "reason": str(e)}

    if traj_file.exists():
        try:
            traj_data = json.loads(traj_file.read_text())
            traj_data.setdefault("info", {})
            traj_data["info"]["submission"] = patch
            traj_data["info"]["verification"] = verification
            traj_file.write_text(json.dumps(traj_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to update trajectory: {e}")

    meta = {
        "instance_id": instance_id,
        "original_instance_id": original_id,
        "model": model or os.environ.get("MSWEA_MODEL_NAME", "default"),
        "timestamp": datetime.now().isoformat(),
        "agent_success": success,
        "has_changes": bool(git_status),
        "patch": patch,
        "verification": verification,
    }
    meta_file.write_text(json.dumps(meta, indent=2))

    cleanup_repo(repo_path)
    return success


def main():
    parser = argparse.ArgumentParser(description="Collect repair trajectories using mini-swe-agent")
    parser.add_argument("--instances", type=Path, default=Path("synthesis/workdir/instances_for_trajectory.jsonl"))
    parser.add_argument("--work-dir", type=Path, default=Path("synthesis/workdir"))
    parser.add_argument("--config", type=Path, default=Path("src/minisweagent/config/extra/swebench_exp.yaml"))
    parser.add_argument("--model", default="", help="Model name (overrides MSWEA_MODEL_NAME env var)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (defaults to work-dir/trajectories)")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-instance timeout in seconds")
    args = parser.parse_args()

    output_dir = args.output or (args.work_dir / "trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    repos_dir = args.work_dir / "repos"

    instances = [json.loads(l) for l in args.instances.read_text().splitlines() if l.strip()]
    logger.info(f"Processing {len(instances)} instances → {output_dir}")

    results = []
    for i, instance in enumerate(instances, 1):
        logger.info(f"\n[{i}/{len(instances)}] {instance['instance_id']}")
        try:
            success = process_instance(instance, repos_dir, output_dir, args.config, args.model, args.timeout)
        except Exception as e:
            logger.error(f"Exception: {e}")
            success = False
        results.append({"instance_id": instance["instance_id"], "success": success})

    successful = sum(r["success"] for r in results)
    summary = {
        "model": args.model or os.environ.get("MSWEA_MODEL_NAME", "default"),
        "total": len(instances),
        "successful": successful,
        "failed": len(instances) - successful,
        "success_rate": successful / len(instances) if instances else 0,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\nDone: {successful}/{len(instances)} succeeded → {output_dir}/summary.json")


if __name__ == "__main__":
    main()
