#!/usr/bin/env python3
"""Audit first-pass cells and rerun only >5% network-contaminated cells."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PYTHON = Path(sys.executable)
FULL_RUNNER = REPO_ROOT / "agent_runner/schedulers/codex_batch.py"
STRICT_RUNNER = REPO_ROOT / "agent_runner/schedulers/codex_strict_batch.py"
SPECS = {
    "full": Path(os.environ.get("TACTILE_FULL_CONFIG", HERE / "full.json")).resolve(),
    "strict": Path(os.environ.get("TACTILE_STRICT_CONFIG", HERE / "strict.json")).resolve(),
}
RUNNERS = {"full": FULL_RUNNER, "strict": STRICT_RUNNER}
STATE_ROOT = Path(
    os.environ.get("TACTILE_STATE_DIR", REPO_ROOT / ".runtime/paper/tactile")
).expanduser().resolve()
NETWORK_RATIO_LIMIT = 0.05
MAX_ATTEMPTS = 8
TERMINAL_VALID = {"success", "finished-no-success"}
INFRA_RE = re.compile(
    r"(?:\b429\b|\b50[234]\b|\b524\b|\b403\b|overload|"
    r"upstream request failed|stream disconnected|timed? out|timeout|"
    r"connection (?:reset|error|refused)|failed to authenticate|"
    r"no available accounts|预扣费额度失败|insufficient (?:fund|balance|credit)|"
    r"quota exceeded)",
    re.IGNORECASE,
)
stop_event = threading.Event()
condition_slots = {"full": threading.Semaphore(2), "strict": threading.Semaphore(2)}
children_lock = threading.Lock()
children: set[subprocess.Popen[Any]] = set()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def base_state(condition: str) -> Path:
    spec = read_json(SPECS[condition])
    root = Path(spec["experiment_root"]).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve() / "batches" / spec["batch_id"] / "batch-state.json"


def controller_events(workspace: Path) -> list[dict[str, Any]]:
    rows = []
    path = workspace / "logs/controller.jsonl"
    if not path.is_file():
        return rows
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def elapsed_seconds(workspace: Path) -> float:
    result = read_json(workspace / ".libero-controller-result.json")
    elapsed = result.get("elapsed_seconds")
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        return float(elapsed)
    times = [float(row["time"]) for row in controller_events(workspace) if isinstance(row.get("time"), (int, float))]
    return max(times) - min(times) if len(times) > 1 else 0.0


def network_estimate(workspace: Path) -> tuple[float, list[str]]:
    stream = workspace / "logs/codex-stream.jsonl"
    samples: list[str] = []
    count = 0
    if stream.is_file():
        for line in stream.open(encoding="utf-8", errors="replace"):
            try:
                row = json.loads(line)
            except Exception:
                continue
            message = ""
            if row.get("type") == "turn.failed":
                message = json.dumps(row.get("error", {}), ensure_ascii=False)
            elif row.get("type") == "error":
                message = str(row.get("message", ""))
            if message and INFRA_RE.search(message):
                count += 1
                if len(samples) < 5:
                    samples.append(message[:240])
    return 30.0 * count, samples


def audit_job(job: dict[str, Any], *, source: str, attempt: int) -> dict[str, Any]:
    reasons: list[str] = []
    state = str(job.get("state") or "unknown")
    if state not in TERMINAL_VALID:
        reasons.append(f"state={state}")
    workspace_text = job.get("workspace")
    workspace = Path(workspace_text) if workspace_text else None
    if workspace is None or not workspace.is_dir():
        reasons.append("missing-workspace")
        return {"valid": False, "reasons": reasons, "state": state, "attempt": attempt, "source": source}
    frames = workspace / "libero-artifacts/frames.jsonl"
    if not frames.is_file() or frames.stat().st_size == 0:
        reasons.append("missing-frames")
    starts = sum(row.get("event") == "controller_started" for row in controller_events(workspace))
    if starts != 1:
        reasons.append(f"controller-starts={starts}")
    result = read_json(workspace / ".libero-controller-result.json")
    if state == "finished-no-success":
        reason = str(job.get("outcome_reason") or result.get("reason") or "")
        if reason != "deadline":
            reasons.append(f"failure-reason={reason or 'missing'}")
    elapsed = elapsed_seconds(workspace)
    network_seconds, samples = network_estimate(workspace)
    ratio = network_seconds / elapsed if elapsed > 0 else 0.0
    return {
        "valid": not reasons,
        "reasons": reasons,
        "state": state,
        "success": state == "success",
        "workspace": str(workspace),
        "attempt": attempt,
        "source": source,
        "elapsed_seconds": elapsed,
        "network_seconds_estimate": network_seconds,
        "network_ratio": ratio,
        "network_estimation_method": "codex-30s-per-explicit-infra-event",
        "infra_samples": samples,
        "timing_valid": ratio <= NETWORK_RATIO_LIMIT,
    }


def rerun_config(condition: str, task: int, seed: int, attempt: int) -> tuple[Path, Path]:
    spec = read_json(SPECS[condition])
    tag = "strict" if condition == "strict" else "full"
    spec["batch_id"] = f"astra-high-libero10-{tag}-t{task:02d}-s{seed:02d}-net-a{attempt:02d}"
    spec["experiment_root"] = str(STATE_ROOT / "network-reruns" / condition)
    spec["cells"] = [{"benchmark": "libero_10", "task_id": task, "seed": seed}]
    spec["max_concurrency"] = 1
    path = STATE_ROOT / "generated-configs" / condition / f"t{task:02d}-s{seed:02d}-a{attempt:02d}.json"
    write_json(path, spec)
    state = Path(spec["experiment_root"]) / "batches" / spec["batch_id"] / "batch-state.json"
    return path, state


def run_scheduler(condition: str, config: Path) -> int:
    process = subprocess.Popen(
        [str(PYTHON), str(RUNNERS[condition]), str(config)],
        cwd=HERE,
        env={
            **os.environ,
            "LIBERO_RUNTIME_DIR": str(
                REPO_ROOT / "libero/runtime" / ("strict" if condition == "strict" else "full")
            ),
        },
        start_new_session=True,
    )
    with children_lock:
        children.add(process)
    try:
        while process.poll() is None and not stop_event.wait(1):
            pass
        if stop_event.is_set() and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return int(process.wait())
    finally:
        with children_lock:
            children.discard(process)


def rerun_cell(condition: str, task: int, seed: int, first: dict[str, Any]) -> dict[str, Any]:
    history = [first]
    selected = first
    with condition_slots[condition]:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if stop_event.is_set():
                break
            config, state_path = rerun_config(condition, task, seed, attempt)
            if state_path.is_file():
                state = read_json(state_path)
                jobs = state.get("jobs", [])
                if len(jobs) == 1 and jobs[0].get("state") in TERMINAL_VALID:
                    report = audit_job(jobs[0], source=str(state_path), attempt=attempt)
                else:
                    report = {"valid": False, "reasons": ["existing-rerun-not-terminal"], "attempt": attempt}
            else:
                code = run_scheduler(condition, config)
                state = read_json(state_path)
                jobs = state.get("jobs", [])
                report = audit_job(jobs[0], source=str(state_path), attempt=attempt) if len(jobs) == 1 else {
                    "valid": False, "reasons": [f"job-count={len(jobs)}"], "attempt": attempt
                }
                report["scheduler_exit_code"] = code
            history.append(report)
            selected = report
            if report.get("valid") and float(report.get("network_ratio", 0.0)) <= NETWORK_RATIO_LIMIT:
                break
    record = {
        "schema_version": 1,
        "status": "complete" if selected.get("valid") and selected.get("timing_valid") else "blocked",
        "condition": condition,
        "task": task,
        "seed": seed,
        "network_ratio_limit": NETWORK_RATIO_LIMIT,
        "selected": selected,
        "history": history,
        "updated_at": now(),
    }
    write_json(STATE_ROOT / "cells" / condition / f"task-{task:02d}" / f"seed-{seed:02d}.json", record)
    return record


def request_stop(*_: Any) -> None:
    stop_event.set()
    with children_lock:
        current = list(children)
    for process in current:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    flagged: list[tuple[str, int, int, dict[str, Any]]] = []
    for condition in ("full", "strict"):
        state_path = base_state(condition)
        state = read_json(state_path)
        if state.get("status") != "completed":
            raise SystemExit(f"first-pass batch is not completed: {state_path}")
        for job in state.get("jobs", []):
            task, seed = int(job["task_id"]), int(job["seed"])
            report = audit_job(job, source=str(state_path), attempt=0)
            record = {
                "schema_version": 1,
                "status": "first-pass-complete" if report.get("valid") else "blocked",
                "condition": condition,
                "task": task,
                "seed": seed,
                "network_ratio_limit": NETWORK_RATIO_LIMIT,
                "selected": report,
                "history": [report],
                "updated_at": now(),
            }
            records[(condition, task, seed)] = record
            write_json(STATE_ROOT / "cells" / condition / f"task-{task:02d}" / f"seed-{seed:02d}.json", record)
            if not report.get("valid") or float(report.get("network_ratio", 0.0)) > NETWORK_RATIO_LIMIT:
                flagged.append((condition, task, seed, report))

    write_json(STATE_ROOT / "network-state.json", {
        "schema_version": 1, "status": "running", "phase": "rerun",
        "network_ratio_limit": NETWORK_RATIO_LIMIT, "flagged_cells": len(flagged),
        "updated_at": now(),
    })
    # Network-cleanup reruns use their own quota: two cells total.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="astra-network-rerun") as executor:
        futures = {
            executor.submit(rerun_cell, condition, task, seed, report): (condition, task, seed)
            for condition, task, seed, report in flagged
        }
        for future in as_completed(futures):
            records[futures[future]] = future.result()

    for key, record in records.items():
        if record.get("status") == "first-pass-complete":
            record["status"] = "complete"
            record["updated_at"] = now()
            condition, task, seed = key
            write_json(STATE_ROOT / "cells" / condition / f"task-{task:02d}" / f"seed-{seed:02d}.json", record)
    blocked = sum(record.get("status") != "complete" for record in records.values())
    for task in range(10):
        for seed in range(5):
            write_json(STATE_ROOT / "pairs" / f"task-{task:02d}" / f"seed-{seed:02d}.json", {
                "schema_version": 1,
                "status": "complete" if all(records[(condition, task, seed)].get("status") == "complete" for condition in ("full", "strict")) else "blocked",
                "task": task,
                "seed": seed,
                "conditions": {condition: records[(condition, task, seed)] for condition in ("full", "strict")},
                "updated_at": now(),
            })
    write_json(STATE_ROOT / "network-state.json", {
        "schema_version": 1,
        "status": "completed" if blocked == 0 else "blocked",
        "phase": "done",
        "network_ratio_limit": NETWORK_RATIO_LIMIT,
        "flagged_cells": len(flagged),
        "blocked_cells": blocked,
        "updated_at": now(),
    })
    return 0 if blocked == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
