#!/usr/bin/env python3
"""Schedule isolated native Codex CLI LIBERO cells at fixed concurrency."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME = REPO_ROOT / "libero/runtime/strict"
CELL = REPO_ROOT / "harness/codex/cell.py"
sys.path.insert(0, str(RUNTIME))

from launch_attempt import prepare_attempt  # noqa: E402
from libero_runtime import atomic_json  # noqa: E402
from review_support import load_object  # noqa: E402


TERMINAL = {"success", "finished-no-success", "controller-error", "cancelled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def frame_count(workspace: str | None) -> int:
    if not workspace:
        return 0
    path = Path(workspace) / "libero-artifacts/frames.jsonl"
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def public_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    spec = load_object(args.config.expanduser().resolve())
    if spec.get("enabled", True) is not True:
        print(f"batch={spec.get('batch_id', 'unknown')} disabled; no cells created")
        return 0
    provider = spec.get("provider", {"id": "openai-chatgpt-auth"})
    provider_id = str(provider.get("id", "openai-chatgpt-auth"))
    if provider_id != "openai-chatgpt-auth":
        required = ("name", "base_url", "env_key", "wire_api")
        missing = [key for key in required if not provider.get(key)]
        if missing:
            raise SystemExit(
                f"custom provider {provider_id!r} is missing: {', '.join(missing)}"
            )
        if not os.environ.get(str(provider["env_key"])):
            raise SystemExit(
                f"custom provider credential is absent: {provider['env_key']}"
            )
    experiment_root = public_path(spec["experiment_root"])
    batch_root = experiment_root / "batches" / str(spec["batch_id"])
    state_path = batch_root / "batch-state.json"
    if args.resume_existing:
        state = load_object(state_path)
        jobs = state["jobs"]
        if not isinstance(jobs, list):
            raise SystemExit("existing batch state has no jobs list")
    else:
        batch_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        atomic_json(batch_root / "batch-plan.json", spec)
        configured_cells = spec.get("cells")
        if configured_cells is None:
            cells = [
                {
                    "benchmark": benchmark,
                    "task_id": task_id,
                    "seed": seed,
                }
                for seed in spec["seeds"]
                for benchmark in spec["benchmarks"]
                for task_id in spec["task_ids"]
            ]
        else:
            if not isinstance(configured_cells, list) or not configured_cells:
                raise SystemExit("cells must be a non-empty list when provided")
            cells = configured_cells
        jobs = [
            {
                "index": index + 1,
                "benchmark": str(cell["benchmark"]),
                "task_id": int(cell["task_id"]),
                "seed": int(cell["seed"]),
                "state": "queued",
                "pid": None,
                "workspace": None,
                "started_at": None,
                "finished_at": None,
                "outcome_reason": None,
                "reviewer": False,
                "review_status": "",
                "retry_count": 0,
                "previous_attempts": [],
            }
            for index, cell in enumerate(cells)
        ]
    children: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    stopped = False

    def request_stop(*_: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not args.resume_existing:
        state = {
            "schema_version": 1,
            "batch_id": spec["batch_id"],
            "status": "running",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "scheduler_pid": os.getpid(),
            "max_concurrency": int(spec["max_concurrency"]),
            "jobs": jobs,
        }
    else:
        state.update(
            status="running",
            scheduler_pid=os.getpid(),
            max_concurrency=int(spec["max_concurrency"]),
            resumed_at=now_iso(),
        )

    retry_limit = int(spec.get("startup_retry_limit", 3))
    retry_delay = float(spec.get("startup_retry_delay_seconds", 60))

    def requeue_startup_failure(job: dict[str, Any], reason: str) -> bool:
        if frame_count(job.get("workspace")) != 0:
            return False
        retry_count = int(job.get("retry_count", 0))
        if retry_count >= retry_limit:
            return False
        history = job.setdefault("previous_attempts", [])
        history.append(
            {
                "workspace": job.get("workspace"),
                "pid": job.get("pid"),
                "started_at": job.get("started_at"),
                "finished_at": now_iso(),
                "reason": reason,
            }
        )
        job.update(
            state="queued",
            pid=None,
            workspace=None,
            started_at=None,
            finished_at=None,
            outcome_reason=None,
            reviewer=False,
            review_status="",
            retry_count=retry_count + 1,
            retry_not_before=time.time() + retry_delay,
        )
        return True

    if args.resume_existing:
        for job in jobs:
            if job.get("state") == "controller-error":
                requeue_startup_failure(job, str(job.get("outcome_reason") or "controller_error"))

    def save() -> None:
        state["updated_at"] = now_iso()
        state["summary"] = dict(Counter(job["state"] for job in jobs))
        atomic_json(state_path, state)

    def launch(job: dict[str, Any]) -> None:
        task_id = int(job["task_id"])
        seed = int(job["seed"])
        benchmark = str(job["benchmark"])
        attempt_id = f"{uuid.uuid4().hex[:7]}-j{job['index']:03d}"
        workspace, config_path, _ = prepare_attempt(
            experiment_root=experiment_root,
            benchmark=benchmark,
            task_id=task_id,
            level=int(spec["observation_level"]),
            attempt_id=attempt_id,
            episode_length=int(spec["episode_length"]),
            bbox_scope=str(spec["bbox_scope"]),
            observation_profile=str(spec["observation_profile"]),
            seed=seed,
            deadline_seconds=int(spec["deadline_seconds"]),
            gateway_url=str(spec["gateway_url"]),
            harness_url="http://127.0.0.1:1",
            provider=provider_id,
            model=str(spec["model"]["name"]),
            agent_preset="codex-cli",
            workflow_variant="plain",
            continue_on_unsuccessful_turn=True,
            active_response_timeout_seconds=None,
            label=f"codex-astra-high-{benchmark}-task{task_id:02d}-seed{seed}-l4-strict",
        )
        cell_home = workspace / ".codex-home"
        cell_home.mkdir(mode=0o700)
        if provider_id == "openai-chatgpt-auth":
            (cell_home / "auth.json").symlink_to(
                Path(spec["codex_auth_file"]).expanduser().resolve()
            )
        config = load_object(config_path)
        config["codex"] = {
            "executable": str(spec["codex_executable"]),
            "home": str(cell_home),
            "model": str(spec["model"]["name"]),
            "reasoning_effort": str(spec["model"]["reasoning_effort"]),
            "context_window": spec["model"].get("context_window"),
            "auto_compact_token_limit": spec["model"].get("auto_compact_token_limit"),
            "provider": provider,
        }
        review_scope = str(spec.get("review", {}).get("scope", "per-task"))
        claim_path = (
            batch_root / "reviewer-claims" / benchmark / f"task-{task_id:02d}.json"
            if review_scope in {"per-task", "first-success-per-task"}
            else batch_root / "reviewer-claim.json"
        )
        config["post_success_review"] = {
            "enabled": False,
            "claim_path": str(claim_path),
            "batch_root": str(batch_root),
            "export_path": str(batch_root / "experience-review"),
            **spec["review"],
        }
        atomic_json(config_path, config)
        launcher_log = workspace / "logs/launcher.log"
        launcher_log.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        handle = launcher_log.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(CELL), str(config_path)],
            cwd=workspace,
            env={**os.environ, "LIBERO_RUNTIME_DIR": str(RUNTIME)},
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        children[process.pid] = (process, handle)
        job.update(
            state="running",
            pid=process.pid,
            workspace=str(workspace),
            started_at=now_iso(),
        )

    save()
    try:
        while not stopped:
            for job in jobs:
                if job["state"] != "running":
                    continue
                pid = int(job["pid"])
                child = children.get(pid)
                if child is not None:
                    if child[0].poll() is None:
                        continue
                    child[1].close()
                    children.pop(pid, None)
                elif process_alive(pid):
                    continue
                result_path = Path(str(job["workspace"])) / ".libero-controller-result.json"
                result = load_object(result_path) if result_path.is_file() else {}
                success = result.get("success") is True
                reason = str(result.get("reason", "controller_missing_result"))
                if reason == "controller_error" and requeue_startup_failure(job, reason):
                    continue
                job.update(
                    state="success"
                    if success
                    else (
                        "controller-error"
                        if reason == "controller_error"
                        else "finished-no-success"
                    ),
                    finished_at=now_iso(),
                    outcome_reason=reason,
                    reviewer=result.get("reviewer", False),
                    review_status=result.get("review_status", "not-selected"),
                    codex_thread_id=result.get("codex_thread_id"),
                )

            running = sum(job["state"] == "running" for job in jobs)
            room = max(0, int(spec["max_concurrency"]) - running)
            eligible = [
                row
                for row in jobs
                if row["state"] == "queued"
                and float(row.get("retry_not_before", 0)) <= time.time()
            ]
            for job in eligible[:room]:
                launch(job)
            save()
            if all(job["state"] in TERMINAL for job in jobs):
                state["status"] = "completed"
                save()
                return 0
            time.sleep(2)
    finally:
        if stopped:
            state["status"] = "stopping"
            save()
            for job in jobs:
                if job["state"] == "running" and job["pid"]:
                    try:
                        os.killpg(int(job["pid"]), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and any(
                job["state"] == "running" and process_alive(int(job["pid"])) for job in jobs
            ):
                time.sleep(0.5)
            for job in jobs:
                if job["state"] == "running":
                    job.update(
                        state="cancelled", finished_at=now_iso(), outcome_reason="stopped"
                    )
            state["status"] = "stopped"
            save()
        for process, handle in children.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
