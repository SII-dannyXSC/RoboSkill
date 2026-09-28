#!/usr/bin/env python3
"""Freeze the paper's paired other-task experience sources by target task."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


BENCHMARK = "libero_10"
PAIR_MAP = {0: 1, 1: 0, 2: 8, 8: 2, 3: 5, 5: 3, 4: 9, 9: 4, 6: 7, 7: 6}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one JSON object")
    return value


def source_seed_map(root: Path) -> dict[tuple[str, int], int]:
    path = root / "source-manifest.json"
    if not path.is_file():
        return {}
    return {
        (str(row["benchmark"]), int(row["task_id"])): int(row["source_seed"])
        for row in load_object(path).get("tasks", [])
    }


def selected_package(root: Path, task_id: int, seeded: bool) -> tuple[Path, int]:
    task_root = root / BENCHMARK / f"task-{task_id:02d}"
    if not task_root.is_dir():
        raise SystemExit(f"missing source task: {task_root}")
    seeds = source_seed_map(root)
    if not seeded:
        if (BENCHMARK, task_id) not in seeds:
            raise SystemExit(f"missing source seed in {root / 'source-manifest.json'}")
        return task_root, seeds[(BENCHMARK, task_id)]
    candidates = sorted(path for path in task_root.glob("seed-*") if path.is_dir())
    if len(candidates) != 1:
        raise SystemExit(f"expected one selected source under {task_root}, got {len(candidates)}")
    try:
        seed = int(candidates[0].name.removeprefix("seed-"))
    except ValueError as error:
        raise SystemExit(f"invalid seed directory: {candidates[0]}") from error
    return candidates[0], seed


def prepare_one(
    name: str, source_root: Path, seeded: bool, output_root: Path
) -> dict[str, Any]:
    destination_root = output_root / name
    destination_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for target_task_id in sorted(PAIR_MAP):
        source_task_id = PAIR_MAP[target_task_id]
        source, seed = selected_package(source_root, source_task_id, seeded)
        task_destination = destination_root / BENCHMARK / f"task-{target_task_id:02d}"
        destination = task_destination / f"seed-{seed}" if seeded else task_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        rows.append(
            {
                "benchmark": BENCHMARK,
                "task_id": target_task_id,
                "source_task_id": source_task_id,
                "source_seed": seed,
                "mapped_package": str(destination),
            }
        )
    manifest = {
        "schema_version": 1,
        "condition": name,
        "pair_map": {str(key): value for key, value in sorted(PAIR_MAP.items())},
        "tasks": rows,
    }
    (destination_root / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--claude-program-root", type=Path, required=True)
    parser.add_argument("--codex-program-root", type=Path, required=True)
    parser.add_argument("--claude-text-action-root", type=Path, required=True)
    parser.add_argument("--codex-text-action-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite mapped sources: {output_root}")
    output_root.mkdir(parents=True)

    specs = {
        "claude-program": (args.claude_program_root, False),
        "codex-program": (args.codex_program_root, True),
        "claude-text-action": (args.claude_text_action_root, False),
        "codex-text-action": (args.codex_text_action_root, True),
    }
    manifests = {
        name: prepare_one(name, root.expanduser().resolve(), seeded, output_root)
        for name, (root, seeded) in specs.items()
    }
    completion = output_root / "mapping-manifest.json"
    completion.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pair_map": {str(k): v for k, v in sorted(PAIR_MAP.items())},
                "conditions": manifests,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"prepared four paired source trees: {completion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
