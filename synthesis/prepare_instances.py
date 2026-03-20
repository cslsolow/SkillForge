#!/usr/bin/env python3
"""
Scan the bugs_from_patch directory and produce a single instances_for_trajectory.jsonl.

Usage:
    python synthesis/prepare_instances.py \\
        --bugs-dir synthesis/workdir/bugs_from_patch \\
        --output synthesis/workdir/instances_for_trajectory.jsonl
"""
import argparse
import json
from pathlib import Path


def find_bug_instances(bugs_dir: Path) -> list:
    instances = []
    if not bugs_dir.exists():
        print(f"⚠ Bugs directory not found: {bugs_dir}")
        return instances

    for instance_dir in sorted(bugs_dir.iterdir()):
        if not instance_dir.is_dir():
            continue
        original_instance_id = instance_dir.name
        print(f"Scanning {original_instance_id}...")

        for test_dir in sorted(instance_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            instances_jsonl = test_dir / "instances.jsonl"
            if not instances_jsonl.exists():
                continue

            count = 0
            for line in instances_jsonl.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    instance = json.loads(line)
                    instance["original_instance_id"] = original_instance_id
                    required = ["instance_id", "base_commit", "problem_statement", "buggy_patch", "FAIL_TO_PASS"]
                    missing = [f for f in required if f not in instance]
                    if missing:
                        print(f"  ⚠ Missing {missing} in {instance.get('instance_id', '?')}")
                        continue
                    instances.append(instance)
                    count += 1
                except json.JSONDecodeError as e:
                    print(f"  ⚠ JSON error: {e}")

            if count:
                print(f"  ✓ {count} instances from {test_dir.name}")

    return instances


def main():
    parser = argparse.ArgumentParser(description="Prepare instances_for_trajectory.jsonl from bugs_from_patch")
    parser.add_argument("--bugs-dir", type=Path, default=Path("synthesis/workdir/bugs_from_patch"))
    parser.add_argument("--output", type=Path, default=Path("synthesis/workdir/instances_for_trajectory.jsonl"))
    args = parser.parse_args()

    instances = find_bug_instances(args.bugs_dir)
    print(f"\nFound {len(instances)} bug instances")

    if not instances:
        print("⚠ No instances found. Run generate_bugs.py first.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")

    print(f"✓ Saved {len(instances)} instances to {args.output}")

    by_orig: dict = {}
    for inst in instances:
        orig = inst.get("original_instance_id", "?")
        by_orig[orig] = by_orig.get(orig, 0) + 1
    print("\nInstances by original ID:")
    for orig_id, cnt in sorted(by_orig.items()):
        print(f"  {orig_id}: {cnt} bugs")


if __name__ == "__main__":
    main()
