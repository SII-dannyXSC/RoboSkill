#!/usr/bin/env python3
"""Trusted wrapper: start timing, launch one agent, then seal the result."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from libero_client import LiberoAPIError, LiberoClient


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _append_history(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    history = directory / ".libero_evaluation_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    os.chmod(history, 0o600)


def _recover_previous_run(client: LiberoClient, marker_path: Path) -> None:
    if not marker_path.exists():
        return
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        run_id = marker["run_id"]
        result_directory = Path(marker["result_directory"])
    except Exception as exc:
        raise SystemExit(
            f"Invalid stale evaluation marker: {marker_path}: {exc}"
        ) from exc
    if not result_directory.is_dir():
        result_directory = Path.cwd() / "recovered_runs" / run_id
        result_directory.mkdir(parents=True, exist_ok=True)
    try:
        recovered = client.finish_evaluation_run(run_id)
    except LiberoAPIError as exc:
        if exc.code not in {"RUN_NOT_FOUND", "RUN_NOT_ACTIVE"}:
            raise SystemExit(
                f"Could not recover previous Evaluation Run {run_id}: {exc}"
            ) from exc
        recovered = {"run_id": run_id, "measurement_status": exc.code.lower()}
    recovered["launcher_recovery"] = {
        "recovered_at": datetime.now(timezone.utc).isoformat(),
        "reason": "previous_launcher_did_not_finish",
    }
    _atomic_json(result_directory / ".libero_evaluation_result.json", recovered)
    _append_history(result_directory, recovered)
    marker_path.unlink(missing_ok=True)
    print(
        f"Recovered previous LIBERO evaluation: {run_id} "
        f"status={recovered.get('measurement_status')}",
        flush=True,
    )


def main() -> int:
    launcher_started_monotonic = time.monotonic()
    launcher_started_at = datetime.now(timezone.utc).isoformat()
    parser = argparse.ArgumentParser(
        description="Create a server-timed LIBERO Run before starting an agent."
    )
    parser.add_argument("--label", default=Path.cwd().name)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="agent command after --"
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an agent command is required after --")

    client = LiberoClient(
        _required("LIBERO_API_URL"),
        _required("LIBERO_LAUNCHER_API_TOKEN"),
    )
    state_directory = Path(
        os.environ.get(
            "LIBERO_EVALUATION_STATE_DIR",
            str(Path.cwd() / ".libero_launcher_state"),
        )
    ).expanduser()
    state_directory.mkdir(parents=True, exist_ok=True)
    os.chmod(state_directory, 0o700)
    identity = os.environ.get("LIBERO_EVALUATION_IDENTITY", args.label)
    safe_identity = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in identity
    )
    lock_path = state_directory / f"{safe_identity}.lock"
    marker_path = state_directory / f"{safe_identity}.active.json"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    os.chmod(lock_path, 0o600)
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(
            f"Another {identity} evaluation launcher is still running."
        ) from exc
    _recover_previous_run(client, marker_path)
    task_id_text = os.environ.get("LIBERO_TASK_ID", "").strip()
    episode_length_text = os.environ.get("LIBERO_EPISODE_LENGTH", "").strip()
    run = client.start_evaluation_run(
        _required("LIBERO_BENCHMARK"),
        task_id=int(task_id_text) if task_id_text else None,
        episode_length=int(episode_length_text) if episode_length_text else None,
        observation_level=int(os.environ.get("LIBERO_OBSERVATION_LEVEL", "1")),
        bbox_scope=os.environ.get("LIBERO_BBOX_SCOPE", "initial"),
        max_attempts=args.max_attempts,
        label=args.label,
        idempotency_key="launch_" + secrets.token_urlsafe(24),
    )
    _atomic_json(
        marker_path,
        {
            "run_id": run["run_id"],
            "label": args.label,
            "result_directory": str(Path.cwd().resolve()),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    child_env = os.environ.copy()
    # Never expose the Run-management credential to the model process.
    child_env.pop("LIBERO_LAUNCHER_API_TOKEN", None)
    child_env["LIBERO_RUN_ID"] = run["run_id"]
    child_env["LIBERO_BENCHMARK"] = run["config"]["benchmark"]
    child_env["LIBERO_TASK_ID"] = str(run["config"]["task_id"])
    child_env["LIBERO_EPISODE_LENGTH"] = str(run["config"]["episode_length"])
    child_env["LIBERO_OBSERVATION_LEVEL"] = str(
        run["config"]["observation_level"]
    )
    child_env["LIBERO_BBOX_SCOPE"] = run["config"]["bbox_scope"]
    print(f"LIBERO evaluation started: {run['run_id']}", flush=True)

    exit_code = 1
    final = run
    agent_started_monotonic = time.monotonic()
    agent_started_at = datetime.now(timezone.utc).isoformat()
    agent_finished_at = None
    agent_process_seconds = None
    try:
        try:
            exit_code = subprocess.run(command, env=child_env, check=False).returncode
        except FileNotFoundError as exc:
            exit_code = 127
            print(
                f"Agent command not found: {exc.filename}. If it is a shell "
                "alias/function, launch it through an interactive login shell.",
                file=sys.stderr,
            )
        agent_process_seconds = time.monotonic() - agent_started_monotonic
        agent_finished_at = datetime.now(timezone.utc).isoformat()
    finally:
        if agent_process_seconds is None:
            agent_process_seconds = time.monotonic() - agent_started_monotonic
            agent_finished_at = datetime.now(timezone.utc).isoformat()
        try:
            final = client.finish_evaluation_run(run["run_id"])
        except LiberoAPIError as exc:
            print(f"Could not finish evaluation Run: {exc}", file=sys.stderr)
            final = client.evaluation_run_status(run["run_id"])

        recorded = dict(final)
        recorded["launcher_timing"] = {
            "launcher_started_at": launcher_started_at,
            "agent_started_at": agent_started_at,
            "agent_finished_at": agent_finished_at,
            "agent_process_seconds": agent_process_seconds,
            "launcher_total_seconds": time.monotonic() - launcher_started_monotonic,
            "agent_exit_code": exit_code,
        }
        output = Path.cwd() / ".libero_evaluation_result.json"
        _atomic_json(output, recorded)
        _append_history(Path.cwd(), recorded)
        marker_path.unlink(missing_ok=True)
        print(
            "LIBERO evaluation finished: "
            f"status={final['measurement_status']} "
            f"time_to_first_success_seconds="
            f"{final.get('time_to_first_success_seconds')}",
            flush=True,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
