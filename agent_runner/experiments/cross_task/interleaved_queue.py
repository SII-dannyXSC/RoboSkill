#!/usr/bin/env python3
"""Run paired Program Prior and Text-Action cells in immediate A/B order."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PYTHON = Path(sys.executable)
STATE_ROOT = Path(
    os.environ.get("CROSS_TASK_STATE_DIR", REPO_ROOT / ".runtime/agent-runner/cross-task")
).expanduser().resolve()
TERMINAL = {"success", "finished-no-success", "controller-error", "cancelled"}

SETTINGS = {
    "claude": {
        "program_config": HERE / "claude-program.json",
        "program_scheduler": REPO_ROOT / "agent_runner/schedulers/claude_program_prior_batch.py",
        "text_action_config": HERE / "claude-text-action.json",
        "text_action_scheduler": REPO_ROOT / "agent_runner/schedulers/claude_text_action_batch.py",
    },
    "codex": {
        "program_config": HERE / "codex-program.json",
        "program_scheduler": REPO_ROOT / "agent_runner/schedulers/codex_program_prior_batch.py",
        "text_action_config": HERE / "codex-text-action.json",
        "text_action_scheduler": REPO_ROOT / "agent_runner/schedulers/codex_text_action_batch.py",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def batch_state_path(spec: dict[str, Any]) -> Path:
    root = Path(spec["experiment_root"]).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return (
        root.resolve()
        / "batches"
        / str(spec["batch_id"])
        / "batch-state.json"
    )


def find_job(state_path: Path, index: int) -> dict[str, Any] | None:
    if not state_path.is_file():
        return None
    matches = [row for row in load(state_path).get("jobs", []) if int(row["index"]) == index]
    if len(matches) != 1:
        raise SystemExit(f"expected one job index {index} in {state_path}")
    return matches[0]


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def queue_state_path(executor: str, worker_id: int, workers: int) -> Path:
    if workers == 1:
        return STATE_ROOT / f"interleaved-queue-state-{executor}.json"
    return STATE_ROOT / (
        f"interleaved-queue-state-{executor}-w{worker_id:02d}-of-{workers:02d}.json"
    )


def write_queue_state(
    executor: str, worker_id: int, workers: int, **updates: Any
) -> None:
    path = queue_state_path(executor, worker_id, workers)
    value = load(path) if path.is_file() else {
        "schema_version": 2,
        "executor": executor,
        "worker_id": worker_id,
        "workers": workers,
    }
    value.update(updates, updated_at=now_iso())
    atomic_write(path, value)


def reconcile_legacy_running(
    executor: str, state_path: Path, worker_id: int, workers: int
) -> None:
    """Adopt Cells left alive after detaching the original scheduler."""

    if not state_path.is_file():
        return
    state = load(state_path)
    for row in state.get("jobs", []):
        if row.get("state") != "running":
            continue
        index = int(row["index"])
        write_queue_state(
            executor,
            worker_id,
            workers,
            status="adopting-existing-cell",
            condition="program",
            job_index=index,
            task_id=row.get("task_id"),
            seed=row.get("seed"),
        )
        pid = int(row["pid"])
        while process_alive(pid):
            time.sleep(2)
        result_path = Path(str(row["workspace"])) / ".libero-controller-result.json"
        result = load(result_path) if result_path.is_file() else {}
        success = result.get("success") is True
        reason = str(result.get("reason", "controller_missing_result"))
        row.update(
            state="success" if success else (
                "controller-error" if reason == "controller_error" else "finished-no-success"
            ),
            finished_at=now_iso(),
            outcome_reason=reason,
            reviewer=result.get("reviewer", False),
            review_status=result.get("review_status", "not-selected"),
        )
        state["updated_at"] = now_iso()
        state["status"] = "migrated-to-interleaved"
        state["summary"] = dict(Counter(job["state"] for job in state["jobs"]))
        atomic_write(state_path, state)


def one_cell_spec(
    base: dict[str, Any], executor: str, condition: str, index: int
) -> tuple[dict[str, Any], Path]:
    task_id = (index - 1) % 10
    seed = (index - 1) // 10
    spec = dict(base)
    spec["task_groups"] = [{"benchmark": "libero_10", "task_ids": [task_id]}]
    spec["seeds"] = [seed]
    spec["batch_id"] = f"{base['batch_id']}-paired-j{index:03d}"
    path = STATE_ROOT / "interleaved-configs" / executor / condition / f"job-{index:03d}.json"
    atomic_write(path, spec)
    return spec, path


def run_one(
    executor: str,
    condition: str,
    index: int,
    worker_id: int,
    workers: int,
) -> None:
    setting = SETTINGS[executor]
    base = load(Path(setting[f"{condition}_config"]))
    legacy = find_job(batch_state_path(base), index)
    if legacy is not None and legacy.get("state") in TERMINAL:
        return

    spec, config_path = one_cell_spec(base, executor, condition, index)
    state_path = batch_state_path(spec)
    current = find_job(state_path, 1)
    if current is not None and current.get("state") in TERMINAL:
        return

    task_id = (index - 1) % 10
    seed = (index - 1) // 10
    write_queue_state(
        executor,
        worker_id,
        workers,
        status="running",
        condition=condition.replace("_", "-"),
        job_index=index,
        task_id=task_id,
        seed=seed,
    )
    command = [
        str(PYTHON),
        str(setting[f"{condition}_scheduler"]),
        str(config_path),
    ]
    if state_path.is_file():
        command.append("--resume-existing")
    subprocess.run(command, check=True)

    finished = find_job(state_path, 1)
    if finished is None or finished.get("state") not in TERMINAL:
        raise SystemExit(f"job did not become terminal: {condition} index={index}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executor", choices=sorted(SETTINGS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-id", type=int, default=0)
    args = parser.parse_args()
    executor = str(args.executor)
    workers = int(args.workers)
    worker_id = int(args.worker_id)
    if workers < 1:
        raise SystemExit("--workers must be positive")
    if worker_id < 0 or worker_id >= workers:
        raise SystemExit("--worker-id must satisfy 0 <= worker-id < workers")
    setting = SETTINGS[executor]

    if worker_id == 0:
        reconcile_legacy_running(
            executor,
            batch_state_path(load(Path(setting["program_config"]))),
            worker_id,
            workers,
        )
        reconcile_legacy_running(
            executor,
            batch_state_path(load(Path(setting["text_action_config"]))),
            worker_id,
            workers,
        )

    for index in range(worker_id + 1, 51, workers):
        run_one(executor, "program", index, worker_id, workers)
        run_one(executor, "text_action", index, worker_id, workers)

    write_queue_state(
        executor,
        worker_id,
        workers,
        status="completed",
        condition=None,
        job_index=None,
        task_id=None,
        seed=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
