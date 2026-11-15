"""Maintain existing repository skills after code or model updates."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path

KINDS = ("keypoints", "env_knowledge")


def resolve_commit(repo: Path | str, ref: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        text=True,
    ).strip()


def _read_at(repo: Path | str, commit: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def _scope_contents(path: str, content: str) -> dict[str, str]:
    from distilling.code_index import parse_file_structure

    structure = parse_file_structure(path, content)
    lines = content.splitlines(keepends=True)
    scopes = {}
    module_lines = set(range(len(lines)))

    def text(start, end):
        return "".join(lines[start - 1 : end])

    for f in structure.functions.values():
        scopes[f"{path}::{f.name}"] = text(f.start_line, f.end_line)
        module_lines.difference_update(range(f.start_line - 1, f.end_line))
    for cls in structure.classes.values():
        class_lines = set(range(cls.start_line - 1, cls.end_line))
        for method in cls.methods.values():
            scopes[f"{path}::{cls.name}::{method.name}"] = text(
                method.start_line, method.end_line
            )
            class_lines.difference_update(range(method.start_line - 1, method.end_line))
        scopes[f"{path}::{cls.name}"] = "".join(lines[i] for i in sorted(class_lines))
        module_lines.difference_update(range(cls.start_line - 1, cls.end_line))
    scopes[path] = "".join(lines[i] for i in sorted(module_lines))
    return scopes


def changed_entities(repo: Path | str, before: str, after: str) -> set[str]:
    """Find changed scopes with the existing source index; fall back to file scope."""
    from distilling.source_files import is_source_file, is_test_file

    before, after = resolve_commit(repo, before), resolve_commit(repo, after)
    paths = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            before,
            after,
        ],
        text=True,
    ).split("\0")
    changed = set()
    for path in filter(None, paths):
        if not is_source_file(path) or is_test_file(path):
            continue
        old, new = _read_at(repo, before, path), _read_at(repo, after, path)
        if old is None or new is None:
            changed.add(path)
            continue
        old_scopes, new_scopes = _scope_contents(path, old), _scope_contents(path, new)
        changed.update(
            k
            for k in old_scopes.keys() | new_scopes.keys()
            if old_scopes.get(k) != new_scopes.get(k)
        )
    return changed


def entity_exists(repo: Path | str, commit: str, entity: str) -> bool:
    path = entity.split("::", 1)[0]
    content = _read_at(repo, commit, path)
    return content is not None and (
        entity == path or entity in _scope_contents(path, content)
    )


def _affected(item: dict, entities: set[str]) -> bool:
    path = item.get("api_path", "")
    return bool(path) and any(
        path == e or path.startswith(e + "::") or e.startswith(path + "::")
        for e in entities
    )


def _stamp(skills: dict, state: dict, model: str, commit: str) -> dict:
    return {
        kind: [
            {
                **item,
                "source_commit": commit,
                "repo_id": state["repo_id"],
                "model": model,
            }
            for item in skills.get(kind, [])
        ]
        for kind in KINDS
    }


def maintain_repository(
    state: dict,
    *,
    changed_entities: set[str],
    new_commit: str,
    synthesize,
    collect,
    distill,
    removed_entities: set[str] | None = None,
) -> dict:
    """Replace affected skills, retaining unrelated records exactly as stored."""
    result = copy.deepcopy(state)
    entities = sorted(changed_entities - (removed_entities or set()))
    instances = (
        synthesize(commit=new_commit, entities=entities, model=state["model"])
        if entities
        else []
    )
    replacements = {}
    if instances:
        trajectories = collect(instances=instances, model=state["model"])
        replacements = _stamp(
            distill(
                instances=instances, trajectories=trajectories, model=state["model"]
            ),
            state,
            state["model"],
            new_commit,
        )
    for kind in KINDS:
        result[kind] = [
            i for i in result.get(kind, []) if not _affected(i, changed_entities)
        ] + [i for i in replacements.get(kind, []) if _affected(i, changed_entities)]
    result["synthetic_instances"] = [*result.get("synthetic_instances", []), *instances]
    result["snapshot_commit"] = new_commit
    return result


def refresh_model(state: dict, *, new_model: str, collect, distill) -> dict:
    """Replay existing synthetic tasks at the fixed snapshot with the new model."""
    result = copy.deepcopy(state)
    instances = [
        i
        for i in result.get("synthetic_instances", [])
        if i.get("base_commit") == state["snapshot_commit"]
    ]
    if not instances:
        raise ValueError("No reusable synthetic instances at the current snapshot")
    trajectories = collect(instances=instances, model=new_model)
    result.update(
        _stamp(
            distill(instances=instances, trajectories=trajectories, model=new_model),
            state,
            new_model,
            state["snapshot_commit"],
        )
    )
    result["model"] = new_model
    return result


def atomic_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as f:
            temporary = Path(f.name)
            json.dump(state, f, indent=2, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_state(state: dict, directory: Path) -> None:
    """Export the existing repo-level JSONL format used by skill injection."""
    directory.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        filename = "repo_" + kind + ".jsonl"
        record = {
            "repo_id": state["repo_id"],
            "kind": kind,
            "items": state.get(kind, []),
        }
        (directory / filename).write_text(
            json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _initialize(args) -> dict:
    if args.state.exists():
        raise FileExistsError(f"State already exists: {args.state}")
    commit = resolve_commit(args.repo, args.commit)
    repo_id = args.repo_id.replace("/", "__")
    instances = []
    for record in _jsonl(args.instances):
        if record.get("repo", "").replace("/", "__") != repo_id:
            continue
        # Import the original synthesis output schema without changing its patches.
        if not record.get("buggy_patch") or not record.get("patch"):
            raise ValueError(
                "Maintenance requires synthetic instances with buggy_patch and patch"
            )
        record["base_commit"] = resolve_commit(args.repo, record["base_commit"])
        instances.append(record)
    state = {
        "repo_id": repo_id,
        "snapshot_commit": commit,
        "model": args.model,
        "synthetic_instances": instances,
    }
    for kind, path in (
        ("keypoints", args.local_skills),
        ("env_knowledge", args.global_skills),
    ):
        state[kind] = [
            copy.deepcopy(item)
            for rec in _jsonl(path)
            if rec.get("repo_id", "").replace("/", "__") == repo_id
            for item in rec.get("items", [])
        ]
    if not instances:
        raise ValueError(f"No synthetic instances found for {repo_id}")
    return state


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "update", "model"):
        p = commands.add_parser(name)
        p.add_argument("--state", type=Path, required=True)
        p.add_argument("--repo", type=Path, required=True)
        p.add_argument("--export-dir", type=Path)
        if name in ("init", "update"):
            p.add_argument("--commit", required=True, help="Repository snapshot commit")
        if name in ("init", "model"):
            p.add_argument("--model", required=True)
        if name == "init":
            p.add_argument(
                "--repo-id", required=True, help="owner/repository or owner__repository"
            )
            p.add_argument("--instances", type=Path, required=True)
            p.add_argument("--local-skills", type=Path, required=True)
            p.add_argument("--global-skills", type=Path, required=True)
        else:
            p.add_argument("--work-dir", type=Path, required=True)
            p.add_argument(
                "--python",
                help="Virtualenv Python with the target repository's test dependencies",
            )
            p.add_argument(
                "--config",
                type=Path,
                default=Path(__file__).parent / "config/extra/synthesis.yaml",
            )
            if name == "update":
                p.add_argument(
                    "--tests",
                    type=Path,
                    help="JSON list of test IDs; otherwise discover existing tests",
                )
    args = parser.parse_args(argv)
    if args.command == "init":
        result = _initialize(args)
    else:
        from synthesis.maintenance_pipeline import MaintenancePipeline

        state = json.loads(args.state.read_text(encoding="utf-8"))
        pipeline = MaintenancePipeline(
            args.repo,
            args.work_dir,
            config=args.config,
            python=args.python,
            repo_id=state["repo_id"],
            tests=json.loads(args.tests.read_text())
            if getattr(args, "tests", None)
            else None,
        )
        if args.command == "model":
            result = refresh_model(
                state,
                new_model=args.model,
                collect=pipeline.collect,
                distill=pipeline.distill,
            )
        else:
            commit = resolve_commit(args.repo, args.commit)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(args.repo),
                    "merge-base",
                    "--is-ancestor",
                    state["snapshot_commit"],
                    commit,
                ],
                check=True,
            )
            entities = changed_entities(args.repo, state["snapshot_commit"], commit)
            removed = {e for e in entities if not entity_exists(args.repo, commit, e)}
            result = maintain_repository(
                state,
                changed_entities=entities,
                removed_entities=removed,
                new_commit=commit,
                synthesize=pipeline.synthesize,
                collect=pipeline.collect,
                distill=pipeline.distill,
            )
    atomic_state(args.state, result)
    if args.export_dir:
        export_state(result, args.export_dir)
    print(f"Skill maintenance state saved to {args.state}")


if __name__ == "__main__":
    main()
