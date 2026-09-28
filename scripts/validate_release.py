#!/usr/bin/env python3
"""Static validation for the source-only public release."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".runtime", ".venv", ".venv312", "third_party", "__pycache__"}
REQUIRED = {
    "libero/src/libero_gateway/app.py",
    "libero/runtime/full/libero_runtime.py",
    "libero/runtime/strict/libero_runtime.py",
    "harness/codex/cell.py",
    "harness/claude/cell.py",
    "agent_runner/run.py",
    "agent_runner/catalog.json",
    "agent_runner/experiments/topk/prepare_libraries.py",
    "scripts/configure_libero.py",
    "scripts/run_libero_gateway.sh",
}


def source_files(suffix: str) -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob(f"*{suffix}")
        if not any(part in SKIP_PARTS or part.startswith(".venv") for part in path.parts)
    )


def main() -> int:
    missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
    if missing:
        raise SystemExit(f"missing required public files: {missing}")

    python_files = source_files(".py")
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    json_files = source_files(".json")
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))

    catalog = json.loads((ROOT / "agent_runner/catalog.json").read_text(encoding="utf-8"))
    if catalog["simulation"]["task_ids"] != list(range(10)):
        raise SystemExit("catalog must define all ten LIBERO-10 tasks")
    tactile = next(item for item in catalog["settings"] if item["id"] == "tactile-ablation")
    if tactile["seeds"] != list(range(5)):
        raise SystemExit("Table 10 tactile seeds must be 0-4")
    if tactile["models"] != ["gpt-6-astra"]:
        raise SystemExit("Table 10 tactile model must be GPT-6 Astra")
    if tactile["conditions"] != ["full", "l4-no-tactile-no-gripper-proprio-v1"]:
        raise SystemExit("Table 10 strict observation profile is incorrect")

    topk = json.loads(
        (ROOT / "agent_runner/configs/topk.example.json").read_text(encoding="utf-8")
    )
    if topk["evaluation_seeds"] != [1, 2, 3, 4]:
        raise SystemExit("same-task Top-K evaluation seeds must be 1-4")
    expected_selections = catalog["agent_selected_top_k"]
    if topk["fixed_selections"] != expected_selections:
        raise SystemExit("Top-K public config does not match the paper selection catalog")
    expected_models = set(expected_selections)
    if set(topk["models"]) != expected_models:
        raise SystemExit("Top-K public config must expose all four paper models")
    for name, model in topk["models"].items():
        expected_executor = "claude" if name in {"opus", "claude-fable-5-1"} else "codex"
        if model["executor"] != expected_executor:
            raise SystemExit(f"Top-K executor mismatch for {name}")
    expected_directories = {
        "gpt-5.6-sol": "gpt-5.6-sol",
        "gpt-6-astra": "gpt-6-astra",
        "opus": "opus-5",
        "claude-fable-5-1": "fable-5.1",
    }
    for name, directory in expected_directories.items():
        if topk["models"][name]["source_directory"] != directory:
            raise SystemExit(f"Top-K source directory mismatch for {name}")

    evolution = json.loads(
        (ROOT / "agent_runner/configs/evolution.example.json").read_text(encoding="utf-8")
    )
    if evolution["evaluation_seeds"] != [100, 101, 102, 103, 104]:
        raise SystemExit("evolution evaluation seeds must be 100-104")
    if evolution["models"]["codex"]["model"] != "gpt-5.6-sol":
        raise SystemExit("evolution Codex model must be GPT-5.6 Sol")
    if evolution["models"]["opus"]["model"] != "opus":
        raise SystemExit("evolution Claude model must be Opus")

    for name in ("codex-program", "codex-text-action"):
        spec = json.loads(
            (ROOT / f"agent_runner/experiments/cross_task/{name}.json").read_text(
                encoding="utf-8"
            )
        )
        if spec["model"]["name"] != "gpt-5.6-sol" or spec["seeds"] != list(range(5)):
            raise SystemExit(f"cross-task Codex protocol mismatch: {name}")
    for name in ("claude-program", "claude-text-action"):
        spec = json.loads(
            (ROOT / f"agent_runner/experiments/cross_task/{name}.json").read_text(
                encoding="utf-8"
            )
        )
        if spec["claude"]["model"] != "opus" or spec["seeds"] != list(range(5)):
            raise SystemExit(f"cross-task Claude protocol mismatch: {name}")

    forbidden = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and (
            path.name == "SKILL.md"
            or ".codex-plugin" in path.parts
            or ".claude-plugin" in path.parts
        )
    ]
    if forbidden:
        raise SystemExit(f"generated skill/plugin files are not publishable: {forbidden}")

    print(
        "release_validation=ok "
        f"python_files={len(python_files)} json_files={len(json_files)} settings={len(catalog['settings'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
