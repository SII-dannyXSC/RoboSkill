#!/usr/bin/env python3
"""Freeze four seed-0 acquisition outputs into the paper Top-K library layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


MODELS = {
    "gpt-5.6-sol": True,
    "gpt-6-astra": True,
    "opus-5": False,
    "fable-5.1": False,
}


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one JSON object")
    return value


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_experience(root: Path) -> None:
    if load_object(root / "validation.json").get("status") != "validated":
        raise SystemExit(f"experience is not validated: {root}")
    index = load_object(root / "program/program-index.json")
    modules = index.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SystemExit(f"experience has no indexed modules: {root}")
    for item in modules:
        path = root / "program/modules" / str(item["filename"])
        if not path.is_file():
            raise SystemExit(f"experience module is missing: {path}")
        expected = item.get("source_sha256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"experience module hash mismatch: {path}")
    if not (root / "episode-summary.md").is_file():
        raise SystemExit(f"experience summary is missing: {root}")


def selected_source(root: Path, task_id: int, seeded: bool) -> tuple[Path, int | None]:
    task = root / f"task-{task_id:02d}"
    if not seeded:
        return task, None
    candidates = sorted(path for path in task.glob("seed-*") if path.is_dir())
    if len(candidates) != 1:
        raise SystemExit(f"expected one selected seed under {task}, got {len(candidates)}")
    try:
        seed = int(candidates[0].name.removeprefix("seed-"))
    except ValueError as error:
        raise SystemExit(f"invalid seed directory: {candidates[0]}") from error
    return candidates[0], seed


def freeze_model(model: str, source_root: Path, output_root: Path) -> list[dict[str, Any]]:
    seeded = MODELS[model]
    rows: list[dict[str, Any]] = []
    for task_id in range(10):
        source, seed = selected_source(source_root, task_id, seeded)
        validate_experience(source)
        destination = output_root / model / f"task-{task_id:02d}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        rows.append(
            {
                "task_id": task_id,
                "source": str(source),
                "source_seed": seed,
                "destination": str(destination),
                "sha256": tree_sha256(destination),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpt-5-6-sol-root", type=Path, required=True)
    parser.add_argument("--gpt-6-astra-root", type=Path, required=True)
    parser.add_argument("--opus-root", type=Path, required=True)
    parser.add_argument("--fable-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite Top-K library: {output_root}")
    output_root.mkdir(parents=True)
    sources = {
        "gpt-5.6-sol": args.gpt_5_6_sol_root,
        "gpt-6-astra": args.gpt_6_astra_root,
        "opus-5": args.opus_root,
        "fable-5.1": args.fable_root,
    }
    models = {
        model: freeze_model(model, path.expanduser().resolve(), output_root)
        for model, path in sources.items()
    }
    manifest = output_root / "library-manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "models": models}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"prepared four Top-K libraries: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
