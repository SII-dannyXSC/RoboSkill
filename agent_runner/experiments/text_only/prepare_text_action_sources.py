#!/usr/bin/env python3
"""Build text + action-trajectory priors from the exact FP source cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent

PACKAGE_FILES = ("metadata.json", "experience.md", "action-schema.json", "trajectory.jsonl", "result.json")
PROPRIOCEPTION_FIELDS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def successful_workspace(workspace: Path) -> None:
    controller = workspace / ".libero-controller-result.json"
    artifact = workspace / "libero-artifacts/result.json"
    values = [load_object(path) for path in (controller, artifact) if path.is_file()]
    if not values or not any(value.get("success") is True for value in values):
        raise RuntimeError(f"source workspace has no authoritative success result: {workspace}")
    if not (workspace / "libero-artifacts/frames.jsonl").is_file():
        raise RuntimeError(f"source workspace has no committed frame index: {workspace}")


def codex_sources_from_batch(batch_root: Path) -> list[tuple[int, int, Path, Path]]:
    """Resolve the one claimed successful review per LIBERO-10 task."""

    review_root = batch_root / "experience-reviews"
    selected: list[tuple[int, int, Path, Path]] = []
    for task_id in range(10):
        task_root = review_root / "libero_10" / f"task-{task_id:02d}"
        reviews = sorted(path for path in task_root.glob("seed-*") if path.is_dir())
        if len(reviews) != 1:
            raise RuntimeError(
                f"expected one Codex source under {task_root}, found {len(reviews)}"
            )
        review = reviews[0]
        source_seed = int(review.name[len("seed-"):])
        claim = load_object(
            batch_root / "reviewer-claims/libero_10" / f"task-{task_id:02d}.json"
        )
        if int(claim["seed"]) != source_seed:
            raise RuntimeError(f"Codex source seed mismatch for task {task_id}")
        workspace = Path(str(claim["workspace"]))
        selected.append((task_id, source_seed, review, workspace))
    return selected


def claude_sources_from_batch(batch_root: Path) -> list[tuple[int, int, Path, Path]]:
    """Resolve the one claimed successful Claude review per LIBERO-10 task."""

    selected: list[tuple[int, int, Path, Path]] = []
    for task_id in range(10):
        review = batch_root / "experience-reviews/libero_10" / f"task-{task_id:02d}"
        claim = load_object(
            batch_root / "reviewer-claims/libero_10" / f"task-{task_id:02d}.json"
        )
        source_seed = int(claim["seed"])
        workspace = Path(str(claim["workspace"]))
        if not review.is_dir():
            raise RuntimeError(f"missing Claude source review: {review}")
        selected.append((task_id, source_seed, review, workspace))
    return selected


def trajectory_rows(workspace: Path) -> list[dict[str, Any]]:
    artifacts = workspace / "libero-artifacts"
    rows: list[dict[str, Any]] = []
    with (artifacts / "frames.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            frame = json.loads(line)
            if not isinstance(frame, dict):
                raise RuntimeError(f"invalid frame at {workspace}:{line_number}")
            action_value: list[float] | None = None
            expected_step: int | None = None
            if frame.get("action"):
                action = load_object(artifacts / str(frame["action"]))
                action_value = action.get("action")
                expected_step = action.get("expected_step_index")
                if not isinstance(action_value, list) or len(action_value) != 7:
                    raise RuntimeError(f"invalid action at {workspace}:{line_number}")
            observation = load_object(artifacts / str(frame["observation"]))
            proprioception = observation.get("proprioception", {})
            if not isinstance(proprioception, dict):
                proprioception = {}
            rows.append(
                {
                    "episode": int(frame["episode_index"]),
                    "step": int(frame["step_index"]),
                    "frame_ref": str(frame["frame_ref"]),
                    "operation": str(frame["operation"]),
                    "action": action_value,
                    "expected_step_index": expected_step,
                    "terminated": frame.get("terminated") is True,
                    "truncated": frame.get("truncated") is True,
                    "reward": frame.get("reward", 0),
                    "proprioception_after": {
                        key: proprioception[key]
                        for key in PROPRIOCEPTION_FIELDS
                        if key in proprioception
                    },
                }
            )
    if not rows or not any(row["action"] is not None for row in rows):
        raise RuntimeError(f"source trajectory contains no actions: {workspace}")
    return rows


def result_summary(workspace: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    result_path = workspace / "libero-artifacts/result.json"
    result = load_object(result_path) if result_path.is_file() else {}
    episodes = sorted({int(row["episode"]) for row in rows})
    return {
        "schema_version": 1,
        "status": result.get("status", "completed"),
        "success": True,
        "steps": result.get("steps"),
        "episodes": episodes,
        "committed_frames": len(rows),
        "committed_actions": sum(row["action"] is not None for row in rows),
    }


def action_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "alignment": "Each step row contains the action issued from the previous committed observation and the resulting proprioception_after.",
        "action_order": ["dx", "dy", "dz", "drot_x", "drot_y", "drot_z", "gripper"],
        "action_range": [-1.0, 1.0],
        "controller": "OSC_POSE",
        "control_delta": True,
        "translation_output_scale_m": 0.05,
        "rotation_output_scale_rad": 0.5,
        "gripper_values": {"-1": "open", "+1": "close"},
        "included_proprioception_after": list(PROPRIOCEPTION_FIELDS),
        "excluded": ["RGB", "depth", "camera calibration", "force/torque", "object ground truth", "program code", "binding", "keyframes", "reasoning logs"],
        "warning": "Historical actions and poses are not current-scene facts. Rebind from the current Reset and do not replay open-loop.",
    }


def build_one(
    destination: Path,
    *,
    executor: str,
    task_id: int,
    source_seed: int,
    review: Path,
    workspace: Path,
) -> dict[str, Any]:
    successful_workspace(workspace)
    summary = review / "episode-summary.md"
    if not summary.is_file() or not summary.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"missing FP episode summary: {summary}")
    rows = trajectory_rows(workspace)
    destination.mkdir(parents=True)
    shutil.copyfile(summary, destination / "experience.md")
    atomic_json(destination / "action-schema.json", action_schema())
    with (destination / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    atomic_json(destination / "result.json", result_summary(workspace, rows))
    metadata = {
        "schema_version": 1,
        "condition": "text-action",
        "executor": executor,
        "benchmark": "libero_10",
        "task_id": task_id,
        "source_seed": source_seed,
        "source_review_sha256": tree_sha256(review),
        "source_frame_index_sha256": sha256_file(workspace / "libero-artifacts/frames.jsonl"),
        "files": {
            name: sha256_file(destination / name)
            for name in PACKAGE_FILES
            if name != "metadata.json"
        },
    }
    atomic_json(destination / "metadata.json", metadata)
    return {
        "benchmark": "libero_10",
        "task_id": task_id,
        "source_seed": source_seed,
        "source_review": str(review),
        "source_workspace": str(workspace),
        "package": str(destination),
        "package_sha256": tree_sha256(destination),
        "committed_frames": len(rows),
        "committed_actions": sum(row["action"] is not None for row in rows),
    }


def validate_existing(root: Path, executor: str) -> bool:
    manifest_path = root / "source-manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = load_object(manifest_path)
    if manifest.get("status") != "validated" or manifest.get("executor") != executor:
        return False
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 10:
        return False
    return all(
        Path(str(row["package"])).is_dir()
        and tree_sha256(Path(str(row["package"]))) == row.get("package_sha256")
        for row in tasks
    )


def build_target(
    target: Path,
    executor: str,
    sources: list[tuple[int, int, Path, Path]],
) -> dict[str, Any]:
    if validate_existing(target, executor):
        return load_object(target / "source-manifest.json")
    if target.exists():
        raise RuntimeError(f"refusing to replace invalid existing source tree: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        rows = []
        for task_id, source_seed, review, workspace in sources:
            suffix = Path("libero_10") / f"task-{task_id:02d}"
            if executor == "codex":
                suffix /= f"seed-{source_seed}"
            row = build_one(
                temporary / suffix,
                executor=executor,
                task_id=task_id,
                source_seed=source_seed,
                review=review,
                workspace=workspace,
            )
            # The temporary tree is atomically renamed below. Persist the final
            # package path so a later validation never points at the vanished
            # staging directory.
            row["package"] = str(target / suffix)
            rows.append(row)
        manifest = {
            "schema_version": 1,
            "status": "validated",
            "condition": "text-action",
            "executor": executor,
            "task_count": len(rows),
            "tasks": rows,
        }
        atomic_json(temporary / "source-manifest.json", manifest)
        os.replace(temporary, target)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-batch-root", type=Path)
    parser.add_argument("--codex-target", type=Path)
    parser.add_argument("--codex-only", action="store_true")
    parser.add_argument("--claude-batch-root", type=Path)
    parser.add_argument("--claude-target", type=Path)
    parser.add_argument("--claude-only", action="store_true")
    args = parser.parse_args()

    if bool(args.codex_batch_root) != bool(args.codex_target):
        parser.error("--codex-batch-root and --codex-target must be provided together")
    if bool(args.claude_batch_root) != bool(args.claude_target):
        parser.error("--claude-batch-root and --claude-target must be provided together")
    if args.codex_only and args.claude_only:
        parser.error("--codex-only and --claude-only are mutually exclusive")
    if not args.claude_only and not args.codex_batch_root:
        parser.error("Codex conversion requires --codex-batch-root and --codex-target")
    if not args.codex_only and not args.claude_batch_root:
        parser.error("Claude conversion requires --claude-batch-root and --claude-target")

    outputs = []
    if not args.claude_only:
        codex_batch_root = args.codex_batch_root.expanduser().resolve()
        codex_target = args.codex_target.expanduser().resolve()
        codex_input = codex_sources_from_batch(codex_batch_root)
        outputs.append(build_target(codex_target, "codex", codex_input))
    if not args.codex_only:
        claude_batch_root = args.claude_batch_root.expanduser().resolve()
        claude_target = args.claude_target.expanduser().resolve()
        claude_input = claude_sources_from_batch(claude_batch_root)
        outputs.append(build_target(claude_target, "claude-code", claude_input))
    print(
        json.dumps(
            {
                "status": "validated",
                "conditions": [
                    {
                        "executor": item["executor"],
                        "tasks": item["task_count"],
                        "frames": sum(row["committed_frames"] for row in item["tasks"]),
                        "actions": sum(row["committed_actions"] for row in item["tasks"]),
                    }
                    for item in outputs
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
