#!/usr/bin/env python3
"""Run one native Claude Code LIBERO cell with a filesystem program prior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME = Path(
    os.environ.get("LIBERO_RUNTIME_DIR", REPO_ROOT / "libero/runtime/full")
).resolve()
sys.path.insert(0, str(RUNTIME))

from libero_runtime import (  # noqa: E402
    GatewayClient,
    GatewayError,
    LeaseProxy,
    append_jsonl,
    atomic_json,
    atomic_text,
    create_lease,
    write_endpoint_file,
)
from prompt_context import render_prompt  # noqa: E402


INITIAL_PROMPT = (
    "立即开始执行系统 Prompt 中定义的 LIBERO 评测。自主读取初始观测、编写并运行控制代码，"
    "持续尝试直到 client.result() 返回 success=true 或外层四小时截止。不要等待确认。\n\n"
    "经验关系：{prior_relation}。本 Workspace 提供了来源任务成功 Seed 产生并通过事实校验的"
    " Program Prior。先读取下面的 program-index.json，再按需读取相关模块：\n"
    "{prior_index}\n\n"
    "Prior 是可复用代码和方法的起点，不是当前场景事实。不要直接复用其中的绝对坐标、深度值、"
    "阈值或动作次数；必须根据当前 Reset 的 RGB、深度、标定、力反馈和本体状态重新计算。"
    "需要修改的模块先复制到 agent/，不要修改 program-prior/。如果经验不适用、执行失败或与"
    "当前观测冲突，可以局部修改、替换模块或完全放弃 Prior，继续自主探索、恢复或 Reset；"
    "不要因为经验或规则问题停止任务。"
)
CONTINUATION_PROMPT = (
    "服务器权威结果仍为 success=false，评测尚未完成。继续当前 Workspace、当前 Claude Session "
    "和当前 LIBERO 环境。先读取 libero-artifacts/frames.jsonl 的最后记录并调用 client.result()，"
    "不要盲目重放可能已经执行的动作；从最新持久化状态继续。只有 success=true 才结束。"
)
REVIEW_PROMPT = """服务器已经权威确认本次 LIBERO 任务成功。你是本批次唯一的经验总结 Agent。

保持当前 Claude Session 和 Workspace，不要调用 client.step() 或 client.reset()，不要继续移动机器人。只复盘你自己这个 task/seed 的真实执行，不读取或汇总其他 seed。以当前对话、agent/、logs/controller.jsonl、libero-artifacts/frames.jsonl 和已经保存的图像为事实来源。

在 experience-review/ 中创建以下文件：
1. action-ledger.json：对象，包含 schema_version=1 和非空 actions。每个 Action 必须有 action_id、purpose、mode、contact_mode、episode、start_step、end_step、method、outcome、observed_effect、failure_mechanism、next_adjustment、reusable_lesson、evidence_refs。Action 是实际发生的高层闭环目标，不是单个环境 Step；不适用的字段用空字符串。episode/start_step/end_step 和 evidence_refs 必须引用 libero-artifacts/frames.jsonl 中真实存在的帧。
2. keyframes.json：对象，包含 schema_version=1 和非空 keyframes。每项有 role、episode、step、frame_ref、reason；frame_ref 必须逐字来自 frames.jsonl。
3. binding.json：把本次场景坐标、阈值和其他数值放入 episode_local_values；把这些参数在新 Reset 场景中应如何重新计算写入 recompute_methods。不要把本次坐标描述成通用事实。
4. episode-summary.md：中文总结成功路径、重要失败、因果修正以及哪些判断由视觉/深度/力反馈/本体感知支持。
5. program/program-index.json 和 program/modules/：从你实际执行过的代码中提取可复用层级 Program。index 包含 schema_version=1、summary、root_node_id、非空 modules 和非空 nodes；modules 每项有 filename、source_path、source_sha256，source_path 指向本 Workspace 的 agent/ 下真实源文件；nodes 每项有 id、purpose、children。可执行节点另有 entrypoint、filename；只负责组织子节点的 composite 节点可以没有 filename，但 children 不能为空。把对应源文件逐字复制到 program/modules/，不要凭空重写未执行的算法。
6. review-result.json：最后写，内容至少为 {"schema_version":1,"status":"complete"}。

不要只在回复中总结；必须写完这些文件。完成后简短回复 experience review complete。
"""
REVIEW_REPAIR_PROMPT = """上一次经验文件没有通过结构/事实引用检查。任务已经成功，仍然不要执行 Step 或 Reset。读取 experience-review/validation-error.json，修复 experience-review/ 中的文件，确保 Action 和关键帧只引用 frames.jsonl 中真实存在的 Episode/Step/frame_ref，最后重写 review-result.json 为 complete。"""


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one object")
    return value


def claim_reviewer(path: Path, payload: dict[str, Any]) -> bool:
    """Atomically let exactly one successful cell become the batch reviewer."""

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


def _review_frames(
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
    """Validate only structural/controller facts, never semantic quality."""

    root = workspace / "experience-review"
    required = [
        "action-ledger.json", "keyframes.json", "binding.json",
        "episode-summary.md", "program/program-index.json", "review-result.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing review files: {missing}")
    result = load_object(root / "review-result.json")
    if result.get("status") != "complete":
        raise ValueError("review-result.json status must be complete")
    action_ledger = load_object(root / "action-ledger.json")
    keyframes = load_object(root / "keyframes.json")
    binding = load_object(root / "binding.json")
    program = load_object(root / "program/program-index.json")
    actions = action_ledger.get("actions")
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
    if not isinstance(modules, list) or not modules or not isinstance(nodes, list) or not nodes:
        raise ValueError("program-index.json requires non-empty modules and nodes")

    real_steps, real_refs, real_frame_keys = _review_frames(workspace)
    action_fields = (
        "action_id", "purpose", "mode", "contact_mode", "episode",
        "start_step", "end_step", "method", "outcome", "observed_effect",
        "failure_mechanism", "next_adjustment", "reusable_lesson", "evidence_refs",
    )
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"actions[{index}] must be an object")
        missing_fields = [field for field in action_fields if field not in action]
        if missing_fields:
            raise ValueError(f"actions[{index}] missing fields: {missing_fields}")
        episode = action.get("episode")
        start = action.get("start_step")
        end = action.get("end_step")
        if not all(isinstance(value, int) for value in (episode, start, end)):
            raise ValueError(f"actions[{index}] requires integer episode/start_step/end_step")
        if start > end or (episode, start) not in real_steps or (episode, end) not in real_steps:
            raise ValueError(f"actions[{index}] references an invalid Step span")
        evidence_refs = action.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not evidence_refs:
            raise ValueError(f"actions[{index}] requires evidence_refs")
        if any(ref not in real_refs for ref in evidence_refs):
            raise ValueError(f"actions[{index}] references an unknown frame_ref")
    for index, mark in enumerate(marks):
        if not isinstance(mark, dict):
            raise ValueError(f"keyframes[{index}] must be an object")
        if any(field not in mark for field in ("role", "episode", "step", "frame_ref", "reason")):
            raise ValueError(f"keyframes[{index}] is missing required fields")
        key = (mark.get("episode"), mark.get("step"), mark.get("frame_ref"))
        if key not in real_frame_keys:
            raise ValueError(f"keyframes[{index}] references an unknown frame")

    module_hashes: dict[str, str] = {}
    for index, module in enumerate(modules):
        if not isinstance(module, dict):
            raise ValueError(f"modules[{index}] must be an object")
        filename = module.get("filename")
        if not isinstance(filename, str) or filename != Path(filename).name or not filename:
            raise ValueError(f"modules[{index}] has unsafe filename")
        source_path = module.get("source_path")
        if not isinstance(source_path, str) or not source_path.startswith("agent/"):
            raise ValueError(f"modules[{index}] requires an agent/ source_path")
        resolved_source = (workspace / source_path).resolve()
        agent_root = (workspace / "agent").resolve()
        if agent_root not in resolved_source.parents or not resolved_source.is_file():
            raise ValueError(f"modules[{index}] source_path is invalid")
        source = root / "program/modules" / filename
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"program module is missing: {filename}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if source.read_bytes() != resolved_source.read_bytes():
            raise ValueError(f"program module differs from source_path: {filename}")
        if module.get("source_sha256") != digest:
            raise ValueError(f"program module hash mismatch: {filename}")
        module_hashes[filename] = digest
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"nodes[{index}] must be an object")
        for field in ("id", "purpose"):
            if not isinstance(node.get(field), str) or not node[field].strip():
                raise ValueError(f"nodes[{index}] requires {field}")
        if node["id"] in node_ids:
            raise ValueError(f"nodes[{index}] duplicates id {node['id']}")
        node_ids.add(node["id"])
        if not isinstance(node.get("children"), list):
            raise ValueError(f"nodes[{index}] requires children list")
        filename = node.get("filename")
        if filename is None:
            if not node["children"]:
                raise ValueError(f"nodes[{index}] without filename must be composite")
        else:
            if filename not in module_hashes:
                raise ValueError(f"nodes[{index}] references an unknown module")
            if not isinstance(node.get("entrypoint"), str) or not node["entrypoint"].strip():
                raise ValueError(f"nodes[{index}] executable node requires entrypoint")
    if program.get("root_node_id") not in node_ids:
        raise ValueError("program root_node_id does not reference a node")
    for index, node in enumerate(nodes):
        unknown_children = [child for child in node["children"] if child not in node_ids]
        if unknown_children:
            raise ValueError(f"nodes[{index}] has unknown children: {unknown_children}")
    if not (root / "episode-summary.md").read_text(encoding="utf-8").strip():
        raise ValueError("episode-summary.md must be non-empty")
    return {
        "schema_version": 1,
        "status": "validated",
        "action_count": len(actions),
        "keyframe_count": len(marks),
        "module_sha256": module_hashes,
    }


def terminate_group(process: subprocess.Popen[Any], grace: float = 15.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_object(config_path)
    workspace = Path(config["workspace"]).expanduser().resolve()
    experiment = config["experiment"]
    claude = config["claude_code"]
    provider = claude.get("provider", {"id": "anthropic-first-party"})
    review = config.get("post_success_review", {"enabled": False})
    review_enabled = bool(review.get("enabled", False))
    prior = config["program_prior"]
    prior_root = Path(prior["path"]).expanduser().resolve()
    target_task_id = int(experiment["task"]["task_id"])
    source_task_id = int(prior.get("source_task_id", target_task_id))
    prior_relation = (
        f"组内另一任务 Task {source_task_id} 的经验，当前目标是 Task {target_task_id}"
        if source_task_id != target_task_id
        else f"当前同一任务 Task {target_task_id} 的经验"
    )
    prior_index = prior_root / "program" / "program-index.json"
    if not prior_index.is_file():
        raise SystemExit(f"validated program prior is missing: {prior_index}")
    deadline_seconds = int(config["deadline_seconds"])
    gateway_url = str(config["gateway_url"]).rstrip("/")
    reviewer_claim_path = (
        Path(review["claim_path"]).expanduser().resolve() if review_enabled else None
    )
    review_export_path = (
        Path(review["export_path"]).expanduser().resolve() if review_enabled else None
    )
    review_grace_seconds = float(review.get("turn_grace_seconds", 90))
    review_timeout_seconds = float(review.get("timeout_seconds", 1800))
    review_max_attempts = int(review.get("max_attempts", 2))

    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    controller_log = logs / "controller.jsonl"
    stream_log = logs / "claude-stream.jsonl"
    stderr_log = logs / "claude-stderr.log"
    endpoint_path = workspace / ".libero-endpoint.json"
    result_path = workspace / ".libero-controller-result.json"
    stop_event = threading.Event()
    post_success_read_only = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    def event(name: str, **details: Any) -> None:
        append_jsonl(controller_log, {"time": time.time(), "event": name, **details})

    started_wall = time.time()
    deadline = time.monotonic() + deadline_seconds
    lease = None
    proxy = None
    process: subprocess.Popen[Any] | None = None
    success = False
    reason = "controller_error"
    launch_count = 0
    exit_codes: list[int] = []
    reviewer_claimed = False
    review_status = "not-selected"
    review_attempts = 0
    review_validation: dict[str, Any] | None = None
    claude_session_id = str(claude["session_id"])
    outcome: dict[str, Any] = {
        "schema_version": 1,
        "started_at": started_wall,
        "success": False,
        "reason": reason,
        "claude_session_id": claude_session_id,
    }

    try:
        event("controller_started", deadline_seconds=deadline_seconds)
        gateway = GatewayClient(gateway_url, stop_event=stop_event)
        lease, _ = create_lease(gateway, workspace, experiment)
        event("libero_session_ready", instruction=lease.instruction)
        initial = lease.mutate("reset", {}, "controller-initial-reset-v1")
        local = initial.get("local_artifacts", {})
        event("initial_reset_persisted", frame_ref=local.get("frame_ref"))

        proxy = LeaseProxy(lease, active=True)
        proxy.set_mutation_check(
            lambda _operation: (
                False,
                "POST_SUCCESS_READ_ONLY",
                "The server-confirmed task is complete; experience review cannot step or reset.",
            )
            if post_success_read_only.is_set()
            else (True, "", "")
        )
        proxy.start()
        write_endpoint_file(workspace, proxy.endpoint)
        event("local_endpoint_ready")

        template = (workspace / "AGENT_PROMPT.md").read_text(encoding="utf-8")
        prompt = render_prompt(
            template,
            experiment,
            instruction=lease.instruction,
            deadline_seconds=deadline_seconds,
            workflow_variant="plain",
        )
        atomic_text(workspace / "PROMPT.md", prompt)
        atomic_json(
            workspace / "libero-evaluation.json",
            {
                "schema_version": 1,
                "created_at": time.time(),
                "instruction": lease.instruction,
                "experiment": experiment,
                "deadline_seconds": deadline_seconds,
                "executor": "claude-code",
                "provider": str(provider.get("id", "anthropic-first-party")),
                "model": claude["model"],
                "effort": claude["effort"],
                "claude_session_id": claude_session_id,
                "credentials_stored": False,
                "remote_identifiers_exposed": False,
                "program_prior": {
                    "path": str(prior_root),
                    "source": prior.get("source"),
                    "snapshot_sha256": prior.get("snapshot_sha256"),
                    "source_seed": prior.get("source_seed"),
                    "source_task_id": source_task_id,
                    "target_task_id": target_task_id,
                    "transfer_class": prior.get("transfer_class"),
                },
            },
        )

        base_command = [
            str(claude["executable"]),
            "--model", str(claude["model"]),
            "--effort", str(claude["effort"]),
            "--autocompact", str(claude.get("autocompact", "auto")),
            "--dangerously-skip-permissions",
            "--safe-mode",
            "--tools", "Bash,Read,Write,Edit",
            "--setting-sources", "project",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
            "--no-chrome",
            "--exclude-dynamic-system-prompt-sections",
            "--append-system-prompt-file", str(workspace / "PROMPT.md"),
            "--output-format", "stream-json",
            "--verbose",
            "-p",
        ]
        heartbeat_at = 0.0
        result_at = 0.0
        backoff = 2.0

        def heartbeat() -> None:
            gateway.request(
                "GET", f"/v2/sessions/{lease.session_id}", token=lease.token
            )

        def finish_current_turn() -> None:
            """Give the successful turn time to finish its response before review."""

            nonlocal process
            if process is None or process.poll() is not None:
                return
            grace_deadline = time.monotonic() + review_grace_seconds
            next_heartbeat = 0.0
            while (
                process.poll() is None
                and not stop_event.is_set()
                and time.monotonic() < grace_deadline
            ):
                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = now + 10
                stop_event.wait(0.5)
            if process.poll() is None:
                event("successful_turn_grace_expired", seconds=review_grace_seconds)
                terminate_group(process)
            code = int(process.wait())
            exit_codes.append(code)
            event("claude_exit", launch_count=launch_count, exit_code=code, stage="task")

        def run_post_success_review(
            stdout_handle: Any, stderr_handle: Any
        ) -> None:
            """Claim and, for one cell only, run a same-session read-only review."""

            nonlocal process, reviewer_claimed, review_status
            nonlocal review_attempts, review_validation
            if not review_enabled:
                review_status = "disabled"
                finish_current_turn()
                return
            assert reviewer_claim_path is not None
            assert review_export_path is not None
            claim = {
                "schema_version": 1,
                "claimed_at": time.time(),
                "task_id": int(experiment["task"]["task_id"]),
                "seed": int(experiment["seed"]),
                "workspace": str(workspace),
                "claude_session_id": claude_session_id,
            }
            reviewer_claimed = claim_reviewer(reviewer_claim_path, claim)
            if not reviewer_claimed:
                review_status = "not-selected"
                event("reviewer_not_selected")
                if process is not None:
                    terminate_group(process)
                return

            review_status = "running"
            post_success_read_only.set()
            event("reviewer_claimed", claim_path=str(reviewer_claim_path))
            finish_current_turn()
            if stop_event.is_set():
                review_status = "stopped"
                return

            review_deadline = time.monotonic() + review_timeout_seconds
            for attempt in range(1, review_max_attempts + 1):
                if stop_event.is_set() or time.monotonic() >= review_deadline:
                    break
                review_attempts = attempt
                review_prompt = REVIEW_PROMPT if attempt == 1 else REVIEW_REPAIR_PROMPT
                command = [
                    *base_command[:-1],
                    "--resume", claude_session_id,
                    "-p", review_prompt,
                ]
                event("review_launch", attempt=attempt)
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=True,
                )
                next_heartbeat = 0.0
                while process.poll() is None and not stop_event.is_set():
                    now = time.monotonic()
                    if now >= review_deadline:
                        event("review_timeout", seconds=review_timeout_seconds)
                        terminate_group(process)
                        break
                    if now >= next_heartbeat:
                        heartbeat()
                        next_heartbeat = now + 10
                    stop_event.wait(0.5)
                if process.poll() is None:
                    terminate_group(process)
                code = int(process.wait())
                exit_codes.append(code)
                event("claude_exit", exit_code=code, stage="review", attempt=attempt)
                try:
                    review_validation = validate_review(workspace)
                except Exception as error:
                    review_status = "validation-failed"
                    atomic_json(
                        workspace / "experience-review" / "validation-error.json",
                        {
                            "schema_version": 1,
                            "attempt": attempt,
                            "error": str(error),
                            "checked_at": time.time(),
                        },
                    )
                    event("review_validation_failed", attempt=attempt, error=str(error))
                    continue

                review_status = "validated"
                atomic_json(
                    workspace / "experience-review" / "validation.json",
                    review_validation,
                )
                review_export_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(workspace / "experience-review", review_export_path)
                event(
                    "review_validated",
                    action_count=review_validation["action_count"],
                    keyframe_count=review_validation["keyframe_count"],
                    export_path=str(review_export_path),
                )
                return

            if stop_event.is_set():
                review_status = "stopped"
            elif time.monotonic() >= review_deadline:
                review_status = "timeout"

        with stream_log.open("a", encoding="utf-8") as stdout_handle, stderr_log.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            while not stop_event.is_set() and time.monotonic() < deadline:
                launch_count += 1
                if launch_count == 1:
                    command = [
                        *base_command[:-1],
                        "--session-id", claude_session_id,
                        "-p", INITIAL_PROMPT.format(
                            prior_index=prior_index, prior_relation=prior_relation
                        ),
                    ]
                else:
                    command = [
                        *base_command[:-1],
                        "--resume", claude_session_id,
                        "-p", CONTINUATION_PROMPT,
                    ]
                event("claude_launch", launch_count=launch_count, resumed=launch_count > 1)
                process = subprocess.Popen(
                    command,
                    cwd=workspace,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=True,
                )

                while process.poll() is None and not stop_event.is_set():
                    now = time.monotonic()
                    if now >= deadline:
                        reason = "deadline"
                        break
                    if now >= heartbeat_at:
                        gateway.request(
                            "GET", f"/v2/sessions/{lease.session_id}", token=lease.token
                        )
                        heartbeat_at = now + 10.0
                    if now >= result_at:
                        try:
                            authoritative = lease.result()
                        except GatewayError as error:
                            if error.status != 409:
                                raise
                        else:
                            if authoritative.get("success") is True:
                                success = True
                                reason = "server_confirmed_success"
                                event("server_confirmed_success")
                                break
                        result_at = now + 2.0
                    stop_event.wait(0.5)

                if success:
                    run_post_success_review(stdout_handle, stderr_handle)
                    break
                if reason == "deadline" or stop_event.is_set():
                    terminate_group(process)
                    break
                code = int(process.wait())
                exit_codes.append(code)
                event("claude_exit", launch_count=launch_count, exit_code=code)
                try:
                    authoritative = lease.result()
                except Exception:
                    authoritative = {"success": False}
                if authoritative.get("success") is True:
                    success = True
                    reason = "server_confirmed_success"
                    event("server_confirmed_success", detected_after_exit=True)
                    run_post_success_review(stdout_handle, stderr_handle)
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "deadline"
                    break
                event("claude_continuation_wait", seconds=min(backoff, remaining))
                stop_event.wait(min(backoff, remaining))
                backoff = min(backoff * 2, 60.0)

        if not success and reason == "controller_error":
            reason = "stopped" if stop_event.is_set() else "deadline"
    except Exception as error:
        reason = "stopped" if stop_event.is_set() else "controller_error"
        outcome["error"] = str(error)
        event("controller_error", error=str(error))
    finally:
        if process is not None:
            terminate_group(process)
        if lease is not None:
            cleanup = GatewayClient(gateway_url, request_timeout=5, retry_initial=0.5, retry_max=2)
            try:
                final_result = cleanup.request(
                    "GET", f"/v2/sessions/{lease.session_id}/result", token=lease.token,
                    max_retry_seconds=20,
                )
                lease.recorder.record_result(final_result)
                outcome["result"] = final_result
                if final_result.get("success") is True:
                    success = True
                    reason = "server_confirmed_success"
            except Exception as error:
                outcome["result_error"] = str(error)
            try:
                cleanup.request(
                    "DELETE", f"/v2/sessions/{lease.session_id}", token=lease.token,
                    max_retry_seconds=20,
                )
            except Exception as error:
                outcome["close_error"] = str(error)
            try:
                cleanup.request(
                    "POST", f"/v2/runs/{lease.run_id}/finish", body={}, token=lease.token,
                    max_retry_seconds=20,
                )
            except Exception as error:
                outcome["finish_error"] = str(error)
        if proxy is not None:
            proxy.close()
        endpoint_path.unlink(missing_ok=True)
        outcome.update(
            finished_at=time.time(),
            elapsed_seconds=time.time() - started_wall,
            success=success,
            reason=reason,
            claude_launch_count=launch_count,
            claude_exit_codes=exit_codes,
            reviewer=reviewer_claimed,
            review_status=review_status,
            review_attempts=review_attempts,
            review_validation=review_validation,
            review_export_path=str(review_export_path) if reviewer_claimed else None,
        )
        atomic_json(result_path, outcome)
        event("controller_finished", success=success, reason=reason)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
