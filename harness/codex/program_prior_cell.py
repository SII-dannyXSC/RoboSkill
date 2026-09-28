#!/usr/bin/env python3
"""Run one native Codex CLI LIBERO cell with a frozen filesystem Program Prior."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
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
from review_support import claim_reviewer, load_object, validate_review  # noqa: E402


INITIAL_PROMPT = (
    "立即开始执行当前 Workspace 中 PROMPT.md 定义的唯一 LIBERO 评测。先读取 PROMPT.md、"
    "AGENT_API.md 和初始 Level-4 Observation；自主探索、编写并运行闭环控制代码，持续尝试直到 "
    "client.result() 返回 success=true 或外层四小时截止。不要等待确认。\n\n"
    "经验关系：{prior_relation}。本 Workspace 提供了来源任务成功执行后产生、并通过结构与事实"
    "引用校验的 Program Prior。先读取下面的 program-index.json，再按需读取相关模块和经验文件：\n"
    "{prior_index}\n\n"
    "Prior 是可复用代码和方法的起点，不是当前场景事实。不得直接复用旧 Episode 的绝对坐标、"
    "深度值、阈值或动作次数；必须根据当前 Reset 的 RGB、深度、标定、力反馈和本体状态重新计算。"
    "需要修改的模块先复制到 agent/，不要修改 program-prior/。如果经验不适用、执行失败或与当前"
    "观测冲突，可以局部修改、替换模块或完全放弃 Prior，并继续自主探索、恢复或 Reset。"
    "不要因为经验或规则问题停止任务。"
)
CONTINUATION_PROMPT = (
    "服务器权威结果仍为 success=false。继续同一个 Workspace、Codex thread 和 LIBERO 环境。"
    "先读取 libero-artifacts/frames.jsonl 最后一条并调用 client.result()，避免盲目重放；从最新"
    "持久化状态继续执行、诊断和恢复。只有 success=true 才结束。"
)
REVIEW_PROMPT = """服务器已经权威确认本次 LIBERO 任务成功。现在由你保存自己这个 task 的经验。

保持当前 Codex thread 和 Workspace，不要调用 client.step() 或 client.reset()，不要继续移动机器人。只复盘你自己这个 task/seed 的真实执行，不读取或汇总其他 seed。以当前 thread、agent/、logs/controller.jsonl、libero-artifacts/frames.jsonl 和已经保存的图像为事实来源。

在 experience-review/ 中创建：
1. action-ledger.json：对象，schema_version=1 和非空 actions。每项必须有 action_id、purpose、mode、contact_mode、episode、start_step、end_step、method、outcome、observed_effect、failure_mechanism、next_adjustment、reusable_lesson、evidence_refs。Action 表示实际发生的高层闭环目标；不适用字段用空字符串。Episode/Step 和 evidence_refs 必须引用 frames.jsonl 中真实帧。
2. keyframes.json：对象，schema_version=1 和非空 keyframes。每项有 role、episode、step、frame_ref、reason，且引用真实帧。
3. binding.json：本场景坐标、阈值等数值放在 episode_local_values；新 Reset 中重新求值的方法放在 recompute_methods，不能把当前坐标写成通用事实。
4. episode-summary.md：中文总结成功路径、关键失败、因果修正，并区分视觉、深度、力反馈和本体感知证据。
5. program/program-index.json 和 program/modules/：从实际执行过的 agent/ 代码提取层级 Program。index 含 schema_version=1、summary、root_node_id、非空 modules 和 nodes。modules 每项有 filename、source_path、source_sha256，source_path 指向 agent/ 下真实文件；nodes 每项有 id、purpose、entrypoint、filename、children。源文件必须逐字复制到 program/modules/，不得凭空改写未执行算法。
6. review-result.json：最后写，至少为 {"schema_version":1,"status":"complete"}。

必须写文件，不能只在回复中总结。完成后简短回复 experience review complete。
"""
REVIEW_REPAIR_PROMPT = """上次经验文件未通过结构或事实引用检查。任务已经成功，仍禁止 Step/Reset。读取 experience-review/validation-error.json，修复经验文件；所有 Episode/Step/frame_ref 必须真实存在，Program 模块必须逐字复制自 agent/ 源文件。最后重写 review-result.json 为 complete。"""


def terminate_group(process: subprocess.Popen[Any], grace: float = 15.0) -> None:
    if process.poll() is not None:
        return
    for sig, timeout in (
        (signal.SIGINT, grace),
        (signal.SIGTERM, 5.0),
        (signal.SIGKILL, 5.0),
    ):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_object(args.config.expanduser().resolve())
    workspace = Path(config["workspace"]).resolve()
    experiment = config["experiment"]
    codex = config["codex"]
    provider = codex.get("provider", {"id": "openai-chatgpt-auth"})
    review = config["post_success_review"]
    review_enabled = review.get("enabled", True) is True
    prior = config["program_prior"]
    prior_root = Path(prior["path"]).resolve()
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
    claim_path = Path(review.get("claim_path", workspace / ".review-disabled-claim.json")).resolve()
    if review.get("batch_root"):
        batch_root = Path(review["batch_root"]).resolve()
    elif claim_path.parent.parent.name == "reviewer-claims":
        batch_root = claim_path.parents[2]
    else:
        batch_root = claim_path.parent
    export_path = (
        batch_root
        / "experience-reviews"
        / str(experiment["task"]["benchmark"])
        / f"task-{int(experiment['task']['task_id']):02d}"
        / f"seed-{int(experiment['seed'])}"
    )
    grace_seconds = float(review.get("turn_grace_seconds", 90))
    review_timeout = float(review.get("timeout_seconds", 1800))
    max_review_attempts = int(review.get("max_attempts", 2))

    logs = workspace / "logs"
    logs.mkdir(parents=True, exist_ok=True, mode=0o700)
    controller_log = logs / "controller.jsonl"
    stream_log = logs / "codex-stream.jsonl"
    stderr_log = logs / "codex-stderr.log"
    endpoint_path = workspace / ".libero-endpoint.json"
    result_path = workspace / ".libero-controller-result.json"
    stop_event = threading.Event()
    read_only = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    def event(name: str, **details: Any) -> None:
        append_jsonl(controller_log, {"time": time.time(), "event": name, **details})

    started = time.time()
    task_deadline = time.monotonic() + deadline_seconds
    lease = None
    proxy = None
    process: subprocess.Popen[Any] | None = None
    thread_id: str | None = None
    success = False
    reason = "controller_error"
    launches = 0
    exit_codes: list[int] = []
    reviewer = False
    review_status = "not-selected"
    review_attempts = 0
    review_validation: dict[str, Any] | None = None
    outcome: dict[str, Any] = {"schema_version": 1, "started_at": started}

    try:
        event("controller_started", deadline_seconds=deadline_seconds)
        gateway = GatewayClient(gateway_url, stop_event=stop_event)
        lease, _ = create_lease(gateway, workspace, experiment)
        event("libero_session_ready", instruction=lease.instruction)
        initial = lease.mutate("reset", {}, "controller-initial-reset-v1")
        event(
            "initial_reset_persisted",
            frame_ref=initial.get("local_artifacts", {}).get("frame_ref"),
        )

        proxy = LeaseProxy(lease, active=True)
        proxy.set_mutation_check(
            lambda _operation: (
                False,
                "POST_SUCCESS_READ_ONLY",
                "Server-confirmed success: review may read files but cannot step or reset.",
            )
            if read_only.is_set()
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
                "executor": "codex-cli-native",
                "provider": str(provider.get("id", "openai-chatgpt-auth")),
                "model": codex["model"],
                "reasoning_effort": codex["reasoning_effort"],
                "imported_experience": True,
                "program_prior": {
                    "path": str(prior_root),
                    "source": prior.get("source"),
                    "snapshot_sha256": prior.get("snapshot_sha256"),
                    "source_seed": prior.get("source_seed"),
                    "source_task_id": source_task_id,
                    "target_task_id": target_task_id,
                    "transfer_class": prior.get("transfer_class"),
                },
                "credentials_stored": False,
            },
        )

        environment = os.environ.copy()
        environment.update(
            {
                "CODEX_HOME": str(codex["home"]),
                "NO_PROXY": "127.0.0.1,localhost",
                "no_proxy": "127.0.0.1,localhost",
            }
        )

        provider_options: list[str] = []
        provider_id = str(provider.get("id", "openai-chatgpt-auth"))
        if provider_id != "openai-chatgpt-auth":
            required = ("name", "base_url", "env_key", "wire_api")
            missing = [key for key in required if not provider.get(key)]
            if missing:
                raise RuntimeError(
                    f"custom Codex provider {provider_id!r} is missing: {', '.join(missing)}"
                )
            env_key = str(provider["env_key"])
            if not environment.get(env_key):
                raise RuntimeError(
                    f"custom Codex provider credential is absent from environment: {env_key}"
                )
            provider_options = [
                "-c",
                f'model_provider="{provider_id}"',
                "-c",
                f'model_providers.{provider_id}.name="{provider["name"]}"',
                "-c",
                f'model_providers.{provider_id}.base_url="{provider["base_url"]}"',
                "-c",
                f'model_providers.{provider_id}.env_key="{env_key}"',
                "-c",
                f'model_providers.{provider_id}.wire_api="{provider["wire_api"]}"',
                "-c",
                "features.shell_snapshot=false",
                "-c",
                "shell_environment_policy.ignore_default_excludes=false",
                "-c",
                f'shell_environment_policy.filters.{env_key}="exclude"',
            ]

        model_metadata_options: list[str] = []
        if codex.get("context_window"):
            model_metadata_options.extend(
                ["-c", f'model_context_window={int(codex["context_window"])}']
            )
        if codex.get("auto_compact_token_limit"):
            model_metadata_options.extend(
                [
                    "-c",
                    f'model_auto_compact_token_limit={int(codex["auto_compact_token_limit"])}',
                ]
            )

        def command_for(text: str) -> list[str]:
            options = [
                "--json",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--ignore-user-config",
                "--dangerously-bypass-approvals-and-sandbox",
                "-m",
                str(codex["model"]),
                *provider_options,
                *model_metadata_options,
                "-c",
                f'model_reasoning_effort="{codex["reasoning_effort"]}"',
                "-c",
                'model_reasoning_summary="detailed"',
                "-c",
                "agents.enabled=false",
                "-c",
                'shell_environment_policy.inherit="all"',
            ]
            if thread_id is None:
                return [str(codex["executable"]), "exec", *options, text]
            return [
                str(codex["executable"]),
                "exec",
                "resume",
                *options,
                thread_id,
                text,
            ]

        def discover_thread_id() -> str | None:
            if not stream_log.is_file():
                return None
            with stream_log.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("type") == "thread.started" and isinstance(
                        row.get("thread_id"), str
                    ):
                        return row["thread_id"]
            return None

        def heartbeat() -> None:
            gateway.request("GET", f"/v2/sessions/{lease.session_id}", token=lease.token)

        def finish_successful_turn() -> None:
            nonlocal process
            if process is None or process.poll() is not None:
                return
            limit = time.monotonic() + grace_seconds
            next_heartbeat = 0.0
            while (
                process.poll() is None
                and not stop_event.is_set()
                and time.monotonic() < limit
            ):
                now = time.monotonic()
                if now >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = now + 10
                stop_event.wait(0.5)
            if process.poll() is None:
                event("successful_turn_grace_expired", seconds=grace_seconds)
                terminate_group(process)
            code = int(process.wait())
            exit_codes.append(code)
            event("codex_exit", exit_code=code, stage="task")

        def post_success_review(stdout_handle: Any, stderr_handle: Any) -> None:
            nonlocal process, reviewer, review_status, review_attempts, review_validation
            nonlocal thread_id
            if thread_id is None:
                thread_id = discover_thread_id()
            if thread_id is None:
                review_status = "missing-thread-id"
                event("review_skipped", reason=review_status)
                return
            if not review_enabled:
                review_status = "disabled"
                event("review_skipped", reason=review_status)
                finish_successful_turn()
                return
            selected = claim_reviewer(
                claim_path,
                {
                    "schema_version": 1,
                    "claimed_at": time.time(),
                    "benchmark": experiment["task"]["benchmark"],
                    "task_id": experiment["task"]["task_id"],
                    "seed": experiment["seed"],
                    "workspace": str(workspace),
                    "codex_thread_id": thread_id,
                },
            )
            if not selected:
                review_status = "not-selected"
                event("review_skipped", reason=review_status, claim_path=str(claim_path))
                finish_successful_turn()
                return
            reviewer = True
            review_status = "running"
            read_only.set()
            event(
                "task_experience_started",
                export_path=str(export_path),
                codex_thread_id=thread_id,
            )
            finish_successful_turn()
            if stop_event.is_set():
                review_status = "stopped"
                return

            deadline = time.monotonic() + review_timeout
            for attempt in range(1, max_review_attempts + 1):
                if stop_event.is_set() or time.monotonic() >= deadline:
                    break
                review_attempts = attempt
                process = subprocess.Popen(
                    command_for(REVIEW_PROMPT if attempt == 1 else REVIEW_REPAIR_PROMPT),
                    cwd=workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=True,
                )
                event("review_launch", attempt=attempt, codex_thread_id=thread_id)
                next_heartbeat = 0.0
                while process.poll() is None and not stop_event.is_set():
                    now = time.monotonic()
                    if now >= deadline:
                        event("review_timeout", seconds=review_timeout)
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
                event("codex_exit", exit_code=code, stage="review", attempt=attempt)
                try:
                    review_validation = validate_review(workspace)
                except Exception as error:
                    review_status = "validation-failed"
                    atomic_json(
                        workspace / "experience-review/validation-error.json",
                        {
                            "schema_version": 1,
                            "attempt": attempt,
                            "checked_at": time.time(),
                            "error": str(error),
                        },
                    )
                    event("review_validation_failed", attempt=attempt, error=str(error))
                    continue
                review_status = "validated"
                atomic_json(
                    workspace / "experience-review/validation.json", review_validation
                )
                export_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(
                    workspace / "experience-review", export_path, dirs_exist_ok=True
                )
                event("review_validated", export_path=str(export_path), **review_validation)
                return
            if stop_event.is_set():
                review_status = "stopped"
            elif time.monotonic() >= deadline:
                review_status = "timeout"

        heartbeat_at = 0.0
        result_at = 0.0
        backoff = 2.0
        with stream_log.open("a", encoding="utf-8") as stdout_handle, stderr_log.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            while not stop_event.is_set() and time.monotonic() < task_deadline:
                launches += 1
                text = (
                    INITIAL_PROMPT.format(
                        prior_index=str(prior_index), prior_relation=prior_relation
                    )
                    if launches == 1
                    else CONTINUATION_PROMPT
                )
                event(
                    "codex_launch",
                    launch_count=launches,
                    resumed=thread_id is not None,
                    codex_thread_id=thread_id,
                )
                process = subprocess.Popen(
                    command_for(text),
                    cwd=workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    start_new_session=True,
                )
                while process.poll() is None and not stop_event.is_set():
                    now = time.monotonic()
                    if now >= task_deadline:
                        reason = "deadline"
                        break
                    if thread_id is None:
                        thread_id = discover_thread_id()
                        if thread_id is not None:
                            outcome["codex_thread_id"] = thread_id
                            event("codex_thread_discovered", codex_thread_id=thread_id)
                    if now >= heartbeat_at:
                        heartbeat()
                        heartbeat_at = now + 10
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
                        result_at = now + 2
                    stop_event.wait(0.5)

                if success:
                    post_success_review(stdout_handle, stderr_handle)
                    break
                if reason == "deadline" or stop_event.is_set():
                    terminate_group(process)
                    break
                code = int(process.wait())
                exit_codes.append(code)
                if thread_id is None:
                    thread_id = discover_thread_id()
                    outcome["codex_thread_id"] = thread_id
                event("codex_exit", exit_code=code, stage="task", launch_count=launches)
                if thread_id is None:
                    raise RuntimeError("Codex CLI exited without a resumable thread id")
                try:
                    authoritative = lease.result()
                except Exception:
                    authoritative = {"success": False}
                if authoritative.get("success") is True:
                    success = True
                    reason = "server_confirmed_success"
                    event("server_confirmed_success", detected_after_exit=True)
                    post_success_review(stdout_handle, stderr_handle)
                    break
                remaining = task_deadline - time.monotonic()
                if remaining <= 0:
                    reason = "deadline"
                    break
                stop_event.wait(min(backoff, remaining))
                backoff = min(backoff * 2, 60)

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
                final = cleanup.request(
                    "GET",
                    f"/v2/sessions/{lease.session_id}/result",
                    token=lease.token,
                    max_retry_seconds=20,
                )
                lease.recorder.record_result(final)
                outcome["result"] = final
                if final.get("success") is True:
                    success = True
                    reason = "server_confirmed_success"
            except Exception as error:
                outcome["result_error"] = str(error)
            try:
                cleanup.request(
                    "DELETE",
                    f"/v2/sessions/{lease.session_id}",
                    token=lease.token,
                    max_retry_seconds=20,
                )
            except Exception as error:
                outcome["close_error"] = str(error)
            try:
                cleanup.request(
                    "POST",
                    f"/v2/runs/{lease.run_id}/finish",
                    body={},
                    token=lease.token,
                    max_retry_seconds=20,
                )
            except Exception as error:
                outcome["finish_error"] = str(error)
        if proxy is not None:
            proxy.close()
        endpoint_path.unlink(missing_ok=True)
        outcome.update(
            finished_at=time.time(),
            elapsed_seconds=time.time() - started,
            success=success,
            reason=reason,
            codex_thread_id=thread_id,
            codex_launch_count=launches,
            codex_exit_codes=exit_codes,
            reviewer=reviewer,
            review_status=review_status,
            review_attempts=review_attempts,
            review_validation=review_validation,
            review_export_path=str(export_path) if reviewer else None,
        )
        atomic_json(result_path, outcome)
        event("controller_finished", success=success, reason=reason)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
