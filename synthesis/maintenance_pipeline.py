"""Maintenance adapters for the existing synthesis, collection and distillation stages."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path


def segment_is_affected(segment: dict, entities: set[str], repo: Path) -> bool:
    from distilling.code_index import parse_file_structure

    path = segment["file"]
    structure = None
    for entity in entities:
        parts = entity.split("::")
        if parts[0] != path:
            continue
        if len(parts) == 1:
            return True
        if structure is None:
            structure = parse_file_structure(
                path, (repo / path).read_text(encoding="utf-8")
            )
        scope = structure.functions.get(parts[1]) or structure.classes.get(parts[1])
        if len(parts) == 3:
            cls = structure.classes.get(parts[1])
            scope = cls.methods.get(parts[2]) if cls else None
        if (
            scope
            and segment["start_line"] <= scope.end_line
            and scope.start_line <= segment["end_line"]
        ):
            return True
    return False


class MaintenancePipeline:
    def __init__(
        self,
        repo: Path,
        work_dir: Path,
        *,
        config: Path | None = None,
        python: str | None = None,
        repo_id: str | None = None,
        tests: list[str] | None = None,
    ):
        self.repo = repo.resolve()
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config = (
            config
            or Path(__file__).parents[1]
            / "src/minisweagent/config/extra/synthesis.yaml"
        ).resolve()
        self.python = str(Path(python or sys.executable).absolute())
        self.repo_id = repo_id
        self.tests = tests

    def _run_dir(self, stage: str) -> Path:
        path = self.work_dir / f"{stage}-{uuid.uuid4().hex}"
        path.mkdir()
        return path

    def _checkout(self, path: Path, commit: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--shared",
                "--no-checkout",
                str(self.repo),
                str(path),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--quiet", "--detach", commit],
            check=True,
        )
        # The existing collector creates local commits for the synthetic bug and repair.
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "SkillForge"], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "skillforge@localhost"],
            check=True,
        )

    def _environment_link(self, parent: Path) -> None:
        prefix = Path(self.python).parent.parent
        if not (prefix / "bin" / "python").is_file():
            raise ValueError("--python must point to a virtualenv's bin/python")
        (parent / "venv").symlink_to(prefix, target_is_directory=True)

    def synthesize(self, *, commit: str, entities: list[str], model: str) -> list[dict]:
        from synthesis.strict_mask_generator import StrictMaskBugGenerator

        affected = set(entities)

        class AffectedGenerator(StrictMaskBugGenerator):
            def _extract_segments_with_context(self, trace_data, logger):
                return [
                    s
                    for s in super()._extract_segments_with_context(trace_data, logger)
                    if segment_is_affected(s, affected, self.repo_path)
                ]

        root = self._run_dir("synthesis")
        snapshot = root / "repos" / "snapshot" / "repo"
        self._checkout(snapshot, commit)
        self._environment_link(snapshot.parent)
        generator = AffectedGenerator(
            str(snapshot), model=model, python_path=self.python
        )
        if self.repo_id:
            generator.repo_owner, generator.repo_name = self.repo_id.replace(
                "/", "__"
            ).split("__", 1)
        tests = (
            self.tests
            if self.tests is not None
            else generator.adapter.discover_tests("snapshot", root / "repos", 120)
        )
        if not tests:
            raise ValueError(
                "No tests available for skill maintenance; supply --tests or check the test environment"
            )
        instances = []
        for test in tests:
            record = generator.generate_bug(
                test_str=test, output_dir=str(root / "bugs")
            )
            if record:
                record = dict(record)
                record["instance_id"] = (
                    f"{self.repo_id or 'maintenance'}-{uuid.uuid4().int}"
                )
                if self.repo_id:
                    record["repo"] = self.repo_id.replace("__", "/")
                instances.append(record)
        (root / "instances.jsonl").write_text(
            "".join(json.dumps(i) + "\n" for i in instances), encoding="utf-8"
        )
        return instances

    def collect(self, *, instances: list[dict], model: str) -> str:
        from synthesis.collect_trajectories import process_instance

        root = self._run_dir("collection")
        output = root / "trajectories"
        output.mkdir()
        repos = root / "repos"
        for record in instances:
            instance = dict(record)
            iid = instance["instance_id"]
            if Path(iid).name != iid or iid in (".", ".."):
                raise ValueError(f"Invalid instance ID: {iid}")
            self._checkout(repos / iid / "repo", instance["base_commit"])
            self._environment_link(repos / iid)
            instance["original_instance_id"] = iid
            process_instance(instance, repos, output, self.config, model, 7200)
            if not (output / iid / f"{iid}_trajectory.json").is_file():
                raise RuntimeError(f"No repair trajectory collected for {iid}")
        return str(output)

    def distill(self, *, instances: list[dict], trajectories: str, model: str) -> dict:
        from distilling.skill_extractor import extract_experiences

        commits = {i["base_commit"] for i in instances}
        if len(commits) != 1:
            raise ValueError(
                "Distill maintenance skills at one fixed repository snapshot"
            )
        root = self._run_dir("distillation")
        snapshot = root / "snapshot"
        self._checkout(snapshot, next(iter(commits)))
        records = root / "instances.jsonl"
        records.write_text(
            "".join(json.dumps(i) + "\n" for i in instances), encoding="utf-8"
        )
        extract_experiences(
            traj_root=trajectories,
            leaderboard_path=records,
            output_dir=root,
            log_path=root / "extraction.log",
            model=model,
            repo_root=str(snapshot),
            extract_keypoints=True,
            extract_env=True,
        )
        result = {}
        expected = {i["instance_id"] for i in instances}
        for kind in ("keypoints", "env_knowledge"):
            records = [
                json.loads(line)
                for line in (root / f"{kind}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if {r["instance_id"] for r in records} != expected:
                raise RuntimeError(
                    f"Incomplete {kind} distillation; maintenance state was not updated"
                )
            result[kind] = [
                {**item, "source_instance_id": rec["instance_id"]}
                for rec in records
                for item in rec.get("items", [])
            ]
        return result
