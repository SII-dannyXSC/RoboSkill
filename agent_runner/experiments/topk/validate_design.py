#!/usr/bin/env python3
"""Validate a public Top-K plan without historical run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from retriever import normalize_tasks, retrieve


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected one JSON object: {path}")
    return value


def public_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def validate_experience(root: Path) -> None:
    validation = load(root / "validation.json")
    if validation.get("status") != "validated":
        raise SystemExit(f"experience is not validated: {root}")
    index = load(root / "program/program-index.json")
    modules = index.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SystemExit(f"experience has no indexed modules: {root}")
    for module in modules:
        path = root / "program/modules" / str(module["filename"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != module["source_sha256"]:
            raise SystemExit(f"module hash mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("model_key")
    parser.add_argument(
        "--check-sources",
        action="store_true",
        help="also validate generated experience packages referenced by the plan",
    )
    args = parser.parse_args()
    cfg = load(args.config.expanduser().resolve())
    if cfg.get("benchmark") != "libero_10":
        raise SystemExit("Top-K publication runner supports benchmark=libero_10")
    if args.model_key not in cfg.get("models", {}):
        raise SystemExit(f"unknown model key: {args.model_key}")
    seeds = [int(seed) for seed in cfg["evaluation_seeds"]]
    if seeds != [1, 2, 3, 4]:
        raise SystemExit(f"unexpected evaluation seeds: {seeds}")
    if set(cfg["tasks"]) != {str(index) for index in range(10)}:
        raise SystemExit("tasks must define all LIBERO-10 task ids")

    tasks = normalize_tasks(cfg["tasks"])
    fixed = cfg.get("fixed_selections", {}).get(args.model_key, {})
    selected: dict[int, list[int]] = {}
    for query in tasks:
        task_id = int(query["task_id"])
        values = fixed.get(str(task_id))
        if values is None:
            hits, _ = retrieve(
                query,
                tasks,
                int(cfg["retrieval"]["top_k"]),
                float(cfg["retrieval"]["threshold"]),
            )
            values = [int(hit.experience_id) for hit in hits]
        ids = [int(value) for value in values]
        if not ids or ids[0] != task_id or len(ids) > int(cfg["retrieval"]["top_k"]):
            raise SystemExit(f"invalid selection for task {task_id}: {ids}")
        selected[task_id] = ids

    checked_sources = 0
    if args.check_sources:
        model = cfg["models"][args.model_key]
        root = public_path(cfg["source_experience_root"]) / str(model["source_directory"])
        for task_id in range(10):
            validate_experience(root / f"task-{task_id:02d}")
            checked_sources += 1

    print(
        json.dumps(
            {
                "status": "validated",
                "model": args.model_key,
                "evaluation_cells": 10 * len(seeds),
                "seeds": seeds,
                "checked_sources": checked_sources,
                "retrieval": selected,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
