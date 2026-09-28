#!/usr/bin/env python3
"""Adaptive scheduler for native Claude Code cells plus one success review."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RUNTIME = REPO_ROOT / "libero/runtime/full"
CELL = REPO_ROOT / "harness/claude/cell.py"
sys.path.insert(0, str(RUNTIME))

from launch_attempt import prepare_attempt  # noqa: E402
from libero_runtime import atomic_json  # noqa: E402


TERMINAL = {"success", "finished-no-success", "controller-error", "cancelled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one object")
    return value


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def active_claude_cells() -> int:
    """Count persistent Claude cell controllers across all experiment batches."""

    try:
        output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum("harness/claude/cell.py" in line for line in output.splitlines())


def frame_count(workspace: str | None) -> int:
    if not workspace:
        return 0
    index = Path(workspace) / "libero-artifacts" / "frames.jsonl"
    if not index.is_file():
        return 0
    with index.open("rb") as handle:
        return sum(1 for _ in handle)


def public_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    spec_path = args.config.expanduser().resolve()
    spec = load_object(spec_path)
    provider = spec.get("provider", {"id": "anthropic-first-party"})
    provider_id = str(provider.get("id", "anthropic-first-party"))
    experiment_root = public_path(spec["experiment_root"])
    batch_root = experiment_root / "batches" / str(spec["batch_id"])
    state_path = batch_root / "batch-state.json"
    if args.resume_existing:
        state = load_object(state_path)
        jobs = state.get("jobs")
        if not isinstance(jobs, list):
            raise SystemExit("existing batch state has no jobs list")
    else:
        batch_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        atomic_json(batch_root / "batch-plan.json", spec)
        jobs: list[dict[str, Any]] = []
        index = 0
        for seed in spec["seeds"]:
            for group in spec["task_groups"]:
                for task_id in group["task_ids"]:
                    index += 1
                    jobs.append(
                        {
                            "index": index,
                            "benchmark": str(group["benchmark"]),
                            "task_id": int(task_id),
                            "seed": int(seed),
                            "state": "queued",
                            "pid": None,
                            "workspace": None,
                            "started_at": None,
                            "finished_at": None,
                            "outcome_reason": None,
                            "retry_count": 0,
                            "previous_attempts": [],
                        }
                    )

    current_limit = int(spec["concurrency"]["initial"])
    maximum = int(spec["concurrency"]["maximum"])
    stable_required = int(spec["concurrency"]["stable_jobs_required"])
    stable_after = float(spec["concurrency"]["stable_after_seconds"])
    children: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    stop = False
    retry_limit = int(spec.get("startup_retry_limit", 3))
    retry_delay = float(spec.get("startup_retry_delay_seconds", 60))

    def handle_stop(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)
    if args.resume_existing:
        state.update(
            status="running",
            resumed_at=now_iso(),
            previous_scheduler_pid=state.get("scheduler_pid"),
            scheduler_pid=os.getpid(),
            current_concurrency=current_limit,
            maximum_concurrency=maximum,
        )
    else:
        state = {
            "schema_version": 1,
            "batch_id": spec["batch_id"],
            "status": "running",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "scheduler_pid": os.getpid(),
            "current_concurrency": current_limit,
            "maximum_concurrency": maximum,
            "ramped_at": None,
            "jobs": jobs,
        }

    def save() -> None:
        state["updated_at"] = now_iso()
        state["current_concurrency"] = current_limit
        state["summary"] = dict(Counter(job["state"] for job in jobs))
        atomic_json(state_path, state)

    def requeue_startup_failure(job: dict[str, Any], reason: str) -> bool:
        if frame_count(job.get("workspace")) != 0:
            return False
        retry_count = int(job.get("retry_count", 0))
        if retry_count >= retry_limit:
            return False
        job["previous_attempts"].append(
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
            retry_count=retry_count + 1,
            retry_not_before=time.time() + retry_delay,
        )
        return True

    def launch(job: dict[str, Any]) -> None:
        benchmark = str(job["benchmark"])
        attempt_id = f"{uuid.uuid4().hex[:7]}-j{job['index']:03d}"
        workspace, config_path, _ = prepare_attempt(
            experiment_root=experiment_root,
            benchmark=benchmark,
            task_id=int(job["task_id"]),
            level=int(spec["observation_level"]),
            attempt_id=attempt_id,
            episode_length=int(spec["episode_length"]),
            bbox_scope=str(spec["bbox_scope"]),
            seed=int(job["seed"]),
            deadline_seconds=int(spec["deadline_seconds"]),
            gateway_url=str(spec["gateway_url"]),
            harness_url="http://127.0.0.1:1",
            provider=provider_id,
            model=str(spec["claude"]["model"]),
            agent_preset="claude-code",
            workflow_variant="plain",
            continue_on_unsuccessful_turn=True,
            active_response_timeout_seconds=None,
            label=(
                f"claude-opus-{benchmark}-task{int(job['task_id']):02d}-"
                f"seed{int(job['seed'])}-l4"
            ),
        )
        config = load_object(config_path)
        config["claude_code"] = {
            **spec["claude"],
            "provider": provider,
            "session_id": str(uuid.uuid4()),
        }
        config["post_success_review"] = {
            "claim_path": str(
                batch_root
                / "reviewer-claims"
                / benchmark
                / f"task-{int(job['task_id']):02d}.json"
            ),
            "export_path": str(
                batch_root
                / "experience-reviews"
                / benchmark
                / f"task-{int(job['task_id']):02d}"
            ),
            **spec["review"],
        }
        atomic_json(config_path, config)
        log_path = workspace / "logs" / "launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(CELL), str(config_path)],
            cwd=workspace,
            env={**os.environ, "LIBERO_RUNTIME_DIR": str(RUNTIME)},
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        children[process.pid] = (process, log_handle)
        job.update(
            state="running",
            pid=process.pid,
            workspace=str(workspace),
            started_at=now_iso(),
        )

    save()
    try:
        while not stop:
            for job in jobs:
                if job["state"] != "running":
                    continue
                pid = int(job["pid"])
                child = children.get(pid)
                if child is not None:
                    code = child[0].poll()
                    if code is None:
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
                    state="success" if success else (
                        "controller-error" if reason == "controller_error" else "finished-no-success"
                    ),
                    finished_at=now_iso(),
                    outcome_reason=reason,
                    reviewer=result.get("reviewer", False),
                    review_status=result.get("review_status", "not-selected"),
                )

            if current_limit < maximum:
                started = [job for job in jobs if job["started_at"] is not None]
                stable = [
                    job for job in started
                    if job["state"] in {"running", "success"} and frame_count(job["workspace"]) >= 2
                ]
                old_enough = False
                if started:
                    first = datetime.fromisoformat(str(started[0]["started_at"]))
                    old_enough = (datetime.now(timezone.utc) - first).total_seconds() >= stable_after
                if old_enough and len(stable) >= stable_required:
                    current_limit = maximum
                    state["ramped_at"] = now_iso()

            running = sum(job["state"] == "running" for job in jobs)
            room = max(0, current_limit - running)
            if spec.get("global_claude_cell_cap") is not None:
                room = min(
                    room,
                    max(0, int(spec["global_claude_cell_cap"]) - active_claude_cells()),
                )
            eligible = [
                job
                for job in jobs
                if job["state"] == "queued"
                and float(job.get("retry_not_before", 0)) <= time.time()
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
        if stop:
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
                    job.update(state="cancelled", finished_at=now_iso(), outcome_reason="stopped")
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
