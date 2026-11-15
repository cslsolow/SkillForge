import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / "code.py").write_text("def f():\n    return 1\n\ndef g():\n    return 2\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Initial")
    return root


def state():
    return {
        "repo_id": "owner__project",
        "snapshot_commit": "old",
        "model": "model-a",
        "synthetic_instances": [{"instance_id": "s1", "base_commit": "old"}],
        "keypoints": [
            {"api_path": "code.py::f", "text": "old"},
            {"api_path": "code.py::g", "text": "keep"},
        ],
        "env_knowledge": [{"api_path": "code.py", "text": "old role"}],
    }


def test_update_replaces_affected_skills_and_preserves_others():
    from minisweagent.maintenance import maintain_repository

    original = state()
    saved = copy.deepcopy(original)
    calls = []

    def synthesize(**kw):
        calls.append(("synthesize", kw))
        return [{"instance_id": "s2", "base_commit": "new"}]

    def collect(**kw):
        calls.append(("collect", kw))
        return "trajectories"

    def distill(**kw):
        calls.append(("distill", kw))
        return {
            "keypoints": [
                {"api_path": "code.py::f", "text": "fresh"},
                {"api_path": "elsewhere.py", "text": "ignore"},
            ],
            "env_knowledge": [{"api_path": "code.py", "text": "fresh role"}],
        }

    result = maintain_repository(
        original,
        changed_entities={"code.py::f"},
        new_commit="new",
        synthesize=synthesize,
        collect=collect,
        distill=distill,
    )
    assert original == saved
    assert [c[0] for c in calls] == ["synthesize", "collect", "distill"]
    assert calls[0][1] == {
        "commit": "new",
        "entities": ["code.py::f"],
        "model": "model-a",
    }
    assert result["keypoints"][0] == saved["keypoints"][1]
    assert result["keypoints"][1]["text"] == "fresh"
    assert result["keypoints"][1]["source_commit"] == "new"
    assert len(result["keypoints"]) == 2
    assert result["env_knowledge"][0]["text"] == "fresh role"
    assert result["snapshot_commit"] == "new"


@pytest.mark.parametrize("stage", ["synthesize", "collect", "distill"])
def test_failed_stage_keeps_input_state(stage):
    from minisweagent.maintenance import maintain_repository

    original = state()
    saved = copy.deepcopy(original)

    def fail(**kw):
        raise RuntimeError("stage failed")

    stages = {
        "synthesize": lambda **kw: [{"instance_id": "new"}],
        "collect": lambda **kw: "trajectory",
        "distill": lambda **kw: {},
    }
    stages[stage] = fail
    with pytest.raises(RuntimeError, match="stage failed"):
        maintain_repository(
            original, changed_entities={"code.py::f"}, new_commit="new", **stages
        )
    assert original == saved


def test_deleted_entities_only_invalidate():
    from minisweagent.maintenance import maintain_repository

    def unexpected(**kw):
        raise AssertionError("deleted entities must not be synthesized")

    result = maintain_repository(
        state(),
        changed_entities={"code.py::f"},
        removed_entities={"code.py::f"},
        new_commit="new",
        synthesize=unexpected,
        collect=unexpected,
        distill=unexpected,
    )
    assert result["keypoints"] == [{"api_path": "code.py::g", "text": "keep"}]
    assert result["env_knowledge"] == []


def test_uncovered_entities_do_not_leave_stale_skills():
    from minisweagent.maintenance import maintain_repository

    def unexpected(**kw):
        raise AssertionError("there are no tasks to collect or distill")

    result = maintain_repository(
        state(),
        changed_entities={"code.py::f"},
        new_commit="new",
        synthesize=lambda **kw: [],
        collect=unexpected,
        distill=unexpected,
    )
    assert result["keypoints"] == [{"api_path": "code.py::g", "text": "keep"}]
    assert result["snapshot_commit"] == "new"


def test_model_refresh_reuses_current_tasks_without_synthesis():
    from minisweagent.maintenance import refresh_model

    original = state()
    original["synthetic_instances"].append(
        {"instance_id": "obsolete", "base_commit": "earlier"}
    )
    saved = copy.deepcopy(original)
    calls = []

    def collect(**kw):
        calls.append(kw)
        return "new trajectories"

    def distill(**kw):
        calls.append(kw)
        return {
            "keypoints": [{"api_path": "code.py::f", "text": "new-model lesson"}],
            "env_knowledge": [],
        }

    result = refresh_model(
        original, new_model="model-b", collect=collect, distill=distill
    )
    assert original == saved
    assert all(c["model"] == "model-b" for c in calls)
    assert all(c["instances"] == saved["synthetic_instances"][:1] for c in calls)
    assert result["snapshot_commit"] == "old"
    assert result["model"] == "model-b"
    assert result["keypoints"][0]["model"] == "model-b"
    assert result["env_knowledge"] == []


def test_model_refresh_rejects_missing_current_tasks():
    from minisweagent.maintenance import refresh_model

    original = state()
    original["snapshot_commit"] = "new"
    with pytest.raises(ValueError, match="snapshot"):
        refresh_model(original, new_model="model-b", collect=None, distill=None)


def test_changed_function_keeps_other_function(repo):
    from minisweagent.maintenance import changed_entities

    before = git(repo, "rev-parse", "HEAD")
    (repo / "code.py").write_text("def f():\n    return 3\n\ndef g():\n    return 2\n")
    git(repo, "commit", "-qam", "Change f")
    assert changed_entities(repo, before, "HEAD") == {"code.py::f"}


def test_changed_method_keeps_other_method(repo):
    from minisweagent.maintenance import changed_entities

    path = repo / "code.py"
    path.write_text(
        "class A:\n    def f(self):\n        return 1\n    def g(self):\n        return 2\n"
    )
    git(repo, "commit", "-qam", "Class")
    before = git(repo, "rev-parse", "HEAD")
    path.write_text(path.read_text().replace("return 1", "return 3"))
    git(repo, "commit", "-qam", "Change method")
    assert changed_entities(repo, before, "HEAD") == {"code.py::A::f"}


def test_module_changes_invalidate_file_scope(repo):
    from minisweagent.maintenance import changed_entities

    before = git(repo, "rev-parse", "HEAD")
    path = repo / "code.py"
    path.write_text("SETTING = 2\n" + path.read_text())
    git(repo, "commit", "-qam", "Setting")
    assert changed_entities(repo, before, "HEAD") == {"code.py"}


def test_deleted_entity_is_detected(repo):
    from minisweagent.maintenance import entity_exists

    assert entity_exists(repo, "HEAD", "code.py::f")
    assert not entity_exists(repo, "HEAD", "code.py::deleted")
    assert not entity_exists(repo, "HEAD", "deleted.py")


def test_init_imports_existing_outputs_and_export_is_agent_readable(repo, tmp_path):
    from minisweagent.maintenance import main
    from minisweagent.skill import load_synthesized_experience_jsonl

    commit = git(repo, "rev-parse", "HEAD")
    instances = tmp_path / "instances.jsonl"
    instances.write_text(
        json.dumps(
            {
                "instance_id": "owner__project-1",
                "repo": "owner/project",
                "base_commit": commit,
                "buggy_patch": "synthetic delta",
                "patch": "synthetic inverse",
                "target_tests": "[]",
            }
        )
        + "\n"
    )
    local = tmp_path / "local.jsonl"
    local.write_text(
        json.dumps(
            {
                "repo_id": "owner__project",
                "kind": "keypoints",
                "items": [
                    {
                        "api_path": "code.py::f",
                        "experience_or_reflection": "lesson",
                        "type": "experience",
                    }
                ],
            }
        )
        + "\n"
    )
    global_ = tmp_path / "global.jsonl"
    global_.write_text(
        json.dumps({"repo_id": "owner__project", "kind": "env_knowledge", "items": []})
        + "\n"
    )
    state_file = tmp_path / "state.json"
    export = tmp_path / "export"
    args = [
        "init",
        "--state",
        str(state_file),
        "--repo",
        str(repo),
        "--commit",
        commit,
        "--model",
        "model-a",
        "--repo-id",
        "owner__project",
        "--instances",
        str(instances),
        "--local-skills",
        str(local),
        "--global-skills",
        str(global_),
        "--export-dir",
        str(export),
    ]
    main(args)
    saved = json.loads(state_file.read_text())
    assert saved["keypoints"][0]["experience_or_reflection"] == "lesson"
    loaded = load_synthesized_experience_jsonl(
        keypoints_path=export / "repo_keypoints.jsonl"
    )
    assert loaded
    assert (
        json.loads((export / "repo_keypoints.jsonl").read_text())["repo_id"]
        == "owner__project"
    )
    with pytest.raises((ValueError, FileExistsError)):
        main(args)


def test_cli_help_does_not_load_llm_clients():
    output = subprocess.check_output(
        [sys.executable, "-m", "minisweagent.maintenance", "--help"], text=True
    )
    assert "init" in output and "update" in output and "model" in output


def test_segment_filter_distinguishes_same_named_methods(repo):
    from synthesis.maintenance_pipeline import segment_is_affected

    (repo / "code.py").write_text(
        "class A:\n    def f(self):\n        pass\n\nclass B:\n    def f(self):\n        pass\n"
    )
    a = {"file": "code.py", "function_name": "f", "start_line": 2, "end_line": 3}
    b = {"file": "code.py", "function_name": "f", "start_line": 6, "end_line": 7}
    assert segment_is_affected(a, {"code.py::A::f"}, repo)
    assert not segment_is_affected(b, {"code.py::A::f"}, repo)
    assert segment_is_affected(b, {"code.py"}, repo)


def test_synthesis_uses_original_generator_with_affected_scope_filter(
    repo, tmp_path, monkeypatch
):
    from synthesis.maintenance_pipeline import MaintenancePipeline
    from synthesis.strict_mask_generator import StrictMaskBugGenerator

    calls = []

    def init(self, repo_path, **kw):
        self.repo_path = Path(repo_path)
        self.base_commit = git(self.repo_path, "rev-parse", "HEAD")

    monkeypatch.setattr(StrictMaskBugGenerator, "__init__", init)
    monkeypatch.setattr(
        StrictMaskBugGenerator,
        "_extract_segments_with_context",
        lambda *a: [
            {"file": "code.py", "start_line": 1, "end_line": 2},
            {"file": "code.py", "start_line": 4, "end_line": 5},
        ],
    )

    def generate(self, **kw):
        segments = self._extract_segments_with_context({}, None)
        calls.append(segments)
        assert len(segments) == 1 and segments[0]["start_line"] == 1
        return {
            "instance_id": "old-id",
            "base_commit": self.base_commit,
            "buggy_patch": "rewrite",
            "patch": "inverse",
            "modified_files": ["code.py"],
        }

    monkeypatch.setattr(StrictMaskBugGenerator, "generate_bug", generate)
    pipeline = MaintenancePipeline(
        repo,
        tmp_path / "work",
        python=sys.executable,
        tests=["tests/test_code.py::test_one", "tests/test_code.py::test_two"],
        repo_id="owner__project",
    )
    commit = git(repo, "rev-parse", "HEAD")
    instances = pipeline.synthesize(
        commit=commit, entities=["code.py::f"], model="model-a"
    )
    assert len(calls) == len(instances) == 2
    assert len({i["instance_id"] for i in instances}) == 2
    assert all(i["base_commit"] == commit for i in instances)
    assert git(repo, "status", "--porcelain") == ""


def test_collection_calls_original_collector_in_isolated_checkouts(
    repo, tmp_path, monkeypatch
):
    from synthesis import collect_trajectories
    from synthesis.maintenance_pipeline import MaintenancePipeline

    commit = git(repo, "rev-parse", "HEAD")
    instance = {
        "instance_id": "owner__project-1",
        "original_instance_id": "old-source",
        "base_commit": commit,
        "buggy_patch": "rewrite",
        "patch": "inverse",
    }
    original = dict(instance)
    calls = []

    def process(record, repos_dir, output_dir, config, model, timeout):
        checkout = repos_dir / record["original_instance_id"] / "repo"
        assert checkout != repo
        assert git(checkout, "rev-parse", "HEAD") == commit
        assert (checkout.parent / "venv" / "bin" / "python").exists()
        (checkout / "code.py").write_text("repair\n")
        calls.append(model)
        out = output_dir / record["instance_id"]
        out.mkdir()
        (out / (record["instance_id"] + "_trajectory.json")).write_text(
            '{"messages": []}'
        )
        return True

    monkeypatch.setattr(collect_trajectories, "process_instance", process)
    pipeline = MaintenancePipeline(repo, tmp_path / "work", python=sys.executable)
    output = pipeline.collect(instances=[instance], model="model-b")
    assert calls == ["model-b"]
    assert Path(output).is_dir()
    assert instance == original
    assert git(repo, "status", "--porcelain") == ""


def test_collection_rejects_missing_trajectory(repo, tmp_path, monkeypatch):
    from synthesis import collect_trajectories
    from synthesis.maintenance_pipeline import MaintenancePipeline

    monkeypatch.setattr(collect_trajectories, "process_instance", lambda *a: False)
    pipeline = MaintenancePipeline(repo, tmp_path / "work", python=sys.executable)
    with pytest.raises(RuntimeError, match="trajectory"):
        pipeline.collect(
            instances=[
                {"instance_id": "s1", "base_commit": git(repo, "rev-parse", "HEAD")}
            ],
            model="model-b",
        )


def test_existing_collector_applies_synthetic_task_and_verifies_repair(
    repo, tmp_path, monkeypatch
):
    from synthesis import collect_trajectories
    from synthesis.maintenance_pipeline import MaintenancePipeline

    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_code.py").write_text(
        "from code import f\n\ndef test_f():\n    assert f() == 1\n"
    )
    # Avoid Python's stdlib module named code when the test runner already imported it.
    (repo / "code.py").rename(repo / "sample.py")
    (tests / "test_code.py").write_text(
        "from sample import f\n\ndef test_f():\n    assert f() == 1\n"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "Tests")
    commit = git(repo, "rev-parse", "HEAD")
    path = repo / "sample.py"
    path.write_text(path.read_text().replace("return 1", "return 0"))
    delta = git(repo, "diff") + "\n"
    inverse = git(repo, "diff", "-R") + "\n"
    git(repo, "checkout", "--", "sample.py")

    def agent(
        repo_path,
        venv_path,
        instance_id,
        problem_statement,
        output_file,
        log_file,
        config_path,
        model,
        timeout,
    ):
        assert "return 0" in (repo_path / "sample.py").read_text()
        assert model == "model-b"
        path = repo_path / "sample.py"
        path.write_text(path.read_text().replace("return 0", "return 1"))
        output_file.write_text(json.dumps({"messages": [], "info": {}}))
        return True

    monkeypatch.setattr(collect_trajectories, "run_agent", agent)
    pipeline = MaintenancePipeline(repo, tmp_path / "work", python=sys.executable)
    output = Path(
        pipeline.collect(
            instances=[
                {
                    "instance_id": "owner__project-1",
                    "base_commit": commit,
                    "buggy_patch": delta,
                    "patch": inverse,
                    "problem_statement": "f returns zero",
                    "target_tests": json.dumps(["tests/test_code.py::test_f"]),
                }
            ],
            model="model-b",
        )
    )
    trajectory = json.loads(
        (output / "owner__project-1" / "owner__project-1_trajectory.json").read_text()
    )
    assert trajectory["info"]["verification"]["all_passed"]
    assert "return 1" in trajectory["info"]["submission"]
    assert git(repo, "status", "--porcelain") == ""


def test_cli_failure_does_not_overwrite_persisted_state(repo, tmp_path, monkeypatch):
    from minisweagent.maintenance import main
    from synthesis.maintenance_pipeline import MaintenancePipeline

    original = state()
    original["snapshot_commit"] = git(repo, "rev-parse", "HEAD")
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(original))
    content = state_file.read_bytes()
    path = repo / "code.py"
    path.write_text(path.read_text().replace("return 1", "return 3"))
    git(repo, "commit", "-qam", "Accepted patch")

    def fail(self, **kw):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(MaintenancePipeline, "synthesize", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        main(
            [
                "update",
                "--state",
                str(state_file),
                "--repo",
                str(repo),
                "--commit",
                "HEAD",
                "--work-dir",
                str(tmp_path / "work"),
            ]
        )
    assert state_file.read_bytes() == content


def test_distillation_uses_existing_schema_and_current_snapshot(
    repo, tmp_path, monkeypatch
):
    from distilling import skill_extractor
    from synthesis.maintenance_pipeline import MaintenancePipeline

    commit = git(repo, "rev-parse", "HEAD")
    tasks = [
        {
            "instance_id": "owner__project-1",
            "base_commit": commit,
            "buggy_patch": "delta",
            "patch": "inverse",
        }
    ]

    def extract(**kw):
        assert kw["model"] == "model-b"
        assert kw["extract_keypoints"] and kw["extract_env"]
        assert git(Path(kw["repo_root"]), "rev-parse", "HEAD") == commit
        assert json.loads(Path(kw["leaderboard_path"]).read_text()) == tasks[0]
        for kind in ("keypoints", "env_knowledge"):
            path = Path(kw["output_dir"]) / (kind + ".jsonl")
            path.write_text(
                json.dumps(
                    {
                        "instance_id": tasks[0]["instance_id"],
                        "kind": kind,
                        "items": [{"api_path": "code.py::f", "text": kind}],
                    }
                )
                + "\n"
            )

    monkeypatch.setattr(skill_extractor, "extract_experiences", extract)
    pipeline = MaintenancePipeline(repo, tmp_path / "work", python=sys.executable)
    skills = pipeline.distill(instances=tasks, trajectories="traces", model="model-b")
    assert skills["keypoints"][0]["source_instance_id"] == "owner__project-1"
    assert skills["env_knowledge"][0]["text"] == "env_knowledge"
