#!/usr/bin/env python3
"""
Clone repositories and create per-instance virtual environments for any SWE-bench subset.

Usage:
    python synthesis/setup_repos.py \\
        --repo-url https://github.com/owner/repo.git \\
        --dataset verified \\
        --filter "owner__repo" \\
        --instances data/instance_ids.txt \\
        --work-dir synthesis/workdir
"""
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Make synthesis package importable for language adapters
sys.path.insert(0, str(Path(__file__).parent))

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-bench",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
}


def load_target_instances(instances_file: Optional[Path]) -> Optional[List[str]]:
    if instances_file is None or not instances_file.exists():
        return None
    return [line.strip() for line in instances_file.read_text().splitlines() if line.strip()]


def load_swebench_data(dataset: str, filter_spec: Optional[str], target_ids: Optional[List[str]]) -> Dict:
    from datasets import load_dataset

    dataset_path = DATASET_MAPPING.get(dataset, dataset)
    logger.info(f"Loading dataset {dataset_path}...")
    ds = load_dataset(dataset_path, split="test")

    data = {}
    for item in ds:
        iid = item["instance_id"]
        if target_ids and iid not in target_ids:
            continue
        if filter_spec and filter_spec not in iid:
            continue
        data[iid] = dict(item)
    logger.info(f"Selected {len(data)} instances")
    return data


def clone_repo(instance_id: str, repo_url: str, base_commit: str, repos_dir: Path) -> Path:
    repo_path = repos_dir / instance_id / "repo"
    if repo_path.exists():
        logger.info(f"Repo exists for {instance_id}, checking out {base_commit[:8]}")
        subprocess.run(["git", "checkout", "-f", base_commit], cwd=repo_path, check=True, capture_output=True)
        return repo_path

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cloning {repo_url} for {instance_id}")
    subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "checkout", "-f", base_commit], cwd=repo_path, check=True, capture_output=True)
    return repo_path


def create_venv(instance_id: str, repo_path: Path, repos_dir: Path, python_bin: str) -> Path:
    venv_path = repos_dir / instance_id / "venv"
    if venv_path.exists():
        logger.info(f"Venv exists for {instance_id}")
        return venv_path

    logger.info(f"Creating venv for {instance_id}")
    subprocess.run([python_bin, "-m", "venv", str(venv_path)], check=True, capture_output=True)

    pip = venv_path / "bin" / "pip"
    subprocess.run([str(pip), "install", "--upgrade", "pip"], check=True, capture_output=True, timeout=300)
    subprocess.run([str(pip), "install", "setuptools", "wheel"], check=True, capture_output=True, timeout=300)
    subprocess.run([str(pip), "install", "-e", str(repo_path)], check=True, capture_output=True, timeout=900)
    subprocess.run(
        [str(pip), "install", "pytest", "pytest-timeout", "coverage"],
        check=True, capture_output=True, timeout=300,
    )
    return venv_path


def save_metadata(instance_id: str, instance_data: Dict, repos_dir: Path):
    metadata_dir = repos_dir / instance_id
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "instance_data.json").write_text(json.dumps(instance_data, indent=2))
    for field, filename in [("test_patch", "test.patch"), ("patch", "fix.patch")]:
        if field in instance_data:
            value = instance_data[field]
            if isinstance(value, str) and not value.startswith("["):
                (metadata_dir / filename).write_text(value)
            else:
                (metadata_dir / filename).write_text(json.dumps(value, indent=2) if isinstance(value, list) else value)


def setup_instance(instance_id: str, instance_data: Dict, repos_dir: Path, repo_url: str, python_bin: str, language: str = "python") -> bool:
    try:
        repo_path = clone_repo(instance_id, repo_url, instance_data["base_commit"], repos_dir)
        if language == "python":
            create_venv(instance_id, repo_path, repos_dir, python_bin)
        else:
            from language_adapters import get_adapter
            adapter = get_adapter(language)
            if not adapter.setup_environment(instance_id, instance_data, repos_dir):
                logger.warning(f"  Environment setup returned False for {instance_id} ({language})")
        save_metadata(instance_id, instance_data, repos_dir)
        logger.info(f"✓ {instance_id}")
        return True
    except Exception as e:
        logger.error(f"✗ {instance_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Setup repos and venvs for SWE-bench instances")
    parser.add_argument("--repo-url", required=True, help="Git URL to clone (e.g. https://github.com/owner/repo.git)")
    parser.add_argument("--dataset", default="verified", help="SWE-bench subset: full/verified/lite or HF path")
    parser.add_argument("--filter", default="", dest="filter_spec", help="Filter instances by substring match on instance_id")
    parser.add_argument("--instances", type=Path, default=None, help="Text file with instance IDs to process (one per line)")
    parser.add_argument("--work-dir", type=Path, default=Path("synthesis/workdir"), help="Working directory")
    parser.add_argument("--python", default=sys.executable, help="Python binary to use for venv creation")
    parser.add_argument("--language", default="python",
                        help="Repository language: python (default), go, typescript")
    args = parser.parse_args()

    repos_dir = args.work_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    target_ids = load_target_instances(args.instances)
    swebench_data = load_swebench_data(args.dataset, args.filter_spec or None, target_ids)

    results = {}
    for instance_id, instance_data in sorted(swebench_data.items()):
        results[instance_id] = setup_instance(instance_id, instance_data, repos_dir, args.repo_url, args.python, language=args.language)

    success = sum(v for v in results.values())
    logger.info(f"\nSetup complete: {success}/{len(results)} successful")
    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.info(f"Failed: {failed}")


if __name__ == "__main__":
    main()
