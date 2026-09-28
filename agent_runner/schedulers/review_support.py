from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def claim_reviewer(path: Path, payload: dict[str, Any]) -> bool:
    """Atomically allow exactly one successful cell to review its own run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def persisted_frames(
    workspace: Path,
) -> tuple[set[tuple[int, int]], set[str], set[tuple[int, int, str]]]:
    steps: set[tuple[int, int]] = set()
    refs: set[str] = set()
    frame_keys: set[tuple[int, int, str]] = set()
    path = workspace / "libero-artifacts" / "frames.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode = row.get("episode_index")
            step = row.get("step_index")
            frame_ref = row.get("frame_ref")
            if isinstance(episode, int) and isinstance(step, int):
                steps.add((episode, step))
            if isinstance(frame_ref, str) and frame_ref:
                refs.add(frame_ref)
                if isinstance(episode, int) and isinstance(step, int):
                    frame_keys.add((episode, step, frame_ref))
    return steps, refs, frame_keys


def validate_review(workspace: Path) -> dict[str, Any]:
    """Validate controller-observable structure and references, not semantics."""

    root = workspace / "experience-review"
    required = [
        "action-ledger.json",
        "keyframes.json",
        "binding.json",
        "episode-summary.md",
        "program/program-index.json",
        "review-result.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing review files: {missing}")

    result = load_object(root / "review-result.json")
    if result.get("status") != "complete":
        raise ValueError("review-result.json status must be complete")
    ledger = load_object(root / "action-ledger.json")
    keyframes = load_object(root / "keyframes.json")
    binding = load_object(root / "binding.json")
    program = load_object(root / "program/program-index.json")
    actions = ledger.get("actions")
    marks = keyframes.get("keyframes")
    modules = program.get("modules")
    nodes = program.get("nodes")
    if not isinstance(actions, list) or not actions:
        raise ValueError("action-ledger.json requires non-empty actions")
    if not isinstance(marks, list) or not marks:
        raise ValueError("keyframes.json requires non-empty keyframes")
    if not isinstance(binding.get("episode_local_values"), dict):
        raise ValueError("binding.json requires episode_local_values object")
    if not isinstance(binding.get("recompute_methods"), dict):
        raise ValueError("binding.json requires recompute_methods object")
    if not isinstance(modules, list) or not modules:
        raise ValueError("program-index.json requires non-empty modules")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("program-index.json requires non-empty nodes")

    real_steps, real_refs, real_frame_keys = persisted_frames(workspace)
    action_fields = (
        "action_id",
        "purpose",
        "mode",
        "contact_mode",
        "episode",
        "start_step",
        "end_step",
        "method",
        "outcome",
        "observed_effect",
        "failure_mechanism",
        "next_adjustment",
        "reusable_lesson",
        "evidence_refs",
    )
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{index}] must be an object")
        missing_fields = [field for field in action_fields if field not in action]
        if missing_fields:
            raise ValueError(f"actions[{index}] missing fields: {missing_fields}")
        episode = action["episode"]
        start = action["start_step"]
        end = action["end_step"]
        if not all(isinstance(value, int) for value in (episode, start, end)):
            raise ValueError(f"actions[{index}] requires integer Episode/Steps")
        if start > end or (episode, start) not in real_steps or (episode, end) not in real_steps:
            raise ValueError(f"actions[{index}] references an invalid Step span")
        evidence = action["evidence_refs"]
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"actions[{index}] requires evidence_refs")
        if any(ref not in real_refs for ref in evidence):
            raise ValueError(f"actions[{index}] references an unknown frame_ref")

    for index, mark in enumerate(marks):
        if not isinstance(mark, dict):
            raise ValueError(f"keyframes[{index}] must be an object")
        fields = ("role", "episode", "step", "frame_ref", "reason")
        if any(field not in mark for field in fields):
            raise ValueError(f"keyframes[{index}] is missing required fields")
        key = (mark["episode"], mark["step"], mark["frame_ref"])
        if key not in real_frame_keys:
            raise ValueError(f"keyframes[{index}] references an unknown frame")

    module_hashes: dict[str, str] = {}
    agent_root = (workspace / "agent").resolve()
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            raise ValueError(f"modules[{index}] must be an object")
        filename = module.get("filename")
        if not isinstance(filename, str) or not filename or filename != Path(filename).name:
            raise ValueError(f"modules[{index}] has unsafe filename")
        source_path = module.get("source_path")
        if not isinstance(source_path, str) or not source_path.startswith("agent/"):
            raise ValueError(f"modules[{index}] requires an agent/ source_path")
        original = (workspace / source_path).resolve()
        if agent_root not in original.parents or not original.is_file():
            raise ValueError(f"modules[{index}] source_path is invalid")
        copy = root / "program/modules" / filename
        if not copy.is_file() or copy.is_symlink():
            raise ValueError(f"program module is missing: {filename}")
        digest = hashlib.sha256(copy.read_bytes()).hexdigest()
        if copy.read_bytes() != original.read_bytes():
            raise ValueError(f"program module differs from source_path: {filename}")
        if module.get("source_sha256") != digest:
            raise ValueError(f"program module hash mismatch: {filename}")
        module_hashes[filename] = digest

    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{index}] must be an object")
        for field in ("id", "purpose", "entrypoint", "filename"):
            if not isinstance(node.get(field), str) or not node[field].strip():
                raise ValueError(f"nodes[{index}] requires {field}")
        if node["filename"] not in module_hashes:
            raise ValueError(f"nodes[{index}] references an unknown module")
        if not isinstance(node.get("children"), list):
            raise ValueError(f"nodes[{index}] requires children list")
        node_ids.add(node["id"])
    if program.get("root_node_id") not in node_ids:
        raise ValueError("root_node_id must identify a real node")
    for index, node in enumerate(nodes):
        if any(child not in node_ids for child in node["children"]):
            raise ValueError(f"nodes[{index}] references an unknown child")
    if not (root / "episode-summary.md").read_text(encoding="utf-8").strip():
        raise ValueError("episode-summary.md must be non-empty")
    return {
        "schema_version": 1,
        "status": "validated",
        "action_count": len(actions),
        "keyframe_count": len(marks),
        "module_sha256": module_hashes,
    }
