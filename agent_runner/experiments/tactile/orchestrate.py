#!/usr/bin/env python3
"""Run paired Full/Strict first passes, then deferred network reruns."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
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
STATE = STATE_ROOT / "suite-state.json"
stop_requested = False
children: dict[str, subprocess.Popen[Any]] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def batch_state_path(spec_path: Path) -> Path:
    spec = read_json(spec_path)
    root = Path(spec["experiment_root"]).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root.resolve() / "batches" / spec["batch_id"] / "batch-state.json"


def process_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False


def save(status: str, phase: str, **details: Any) -> None:
    write_json(
        STATE,
        {
            "schema_version": 1,
            "status": status,
            "phase": phase,
            "updated_at": now(),
            "model": "gpt-6-astra",
            "reasoning_effort": "high",
            "network_ratio_limit": 0.05,
            "first_pass_concurrency": {"full": 2, "strict": 2, "total": 4},
            **details,
        },
    )


def status_of(spec_path: Path) -> tuple[str | None, dict[str, Any]]:
    path = batch_state_path(spec_path)
    if not path.is_file():
        return None, {}
    value = read_json(path)
    return str(value.get("status")), value


def start_or_adopt(condition: str) -> None:
    status, state = status_of(SPECS[condition])
    if status == "completed":
        return
    if status == "running" and process_alive(state.get("scheduler_pid")):
        return
    command = [str(PYTHON), str(RUNNERS[condition]), str(SPECS[condition])]
    if status is not None:
        command.append("--resume-existing")
    runtime = REPO_ROOT / "libero/runtime" / ("strict" if condition == "strict" else "full")
    children[condition] = subprocess.Popen(
        command,
        cwd=HERE,
        env={**os.environ, "LIBERO_RUNTIME_DIR": str(runtime)},
        start_new_session=True,
    )


def request_stop(*_: Any) -> None:
    global stop_requested
    stop_requested = True
    for child in list(children.values()):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def wait_first_pass() -> None:
    while not stop_requested:
        statuses = {condition: status_of(spec)[0] for condition, spec in SPECS.items()}
        save("running", "first-pass", conditions=statuses)
        if all(value == "completed" for value in statuses.values()):
            return
        for condition, child in list(children.items()):
            code = child.poll()
            if code is not None and code != 0:
                raise RuntimeError(f"{condition} first-pass scheduler exited {code}")
        time.sleep(15)
    raise RuntimeError("suite stopped")


def main() -> int:
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    for path in (PYTHON, FULL_RUNNER, STRICT_RUNNER):
        if not path.exists():
            raise SystemExit(f"required executable is missing: {path}")
    save("running", "startup")
    for condition in ("full", "strict"):
        start_or_adopt(condition)
    try:
        wait_first_pass()
        save("running", "network-audit")
        code = subprocess.call([str(PYTHON), str(HERE / "network_rerun.py")], cwd=HERE)
        if code != 0:
            save("blocked", "network-rerun", returncode=code)
            return code
        save("completed", "done")
        return 0
    except Exception as error:
        save("stopped" if stop_requested else "failed", "first-pass", error=str(error))
        return 130 if stop_requested else 1
    finally:
        request_stop()


if __name__ == "__main__":
    raise SystemExit(main())
