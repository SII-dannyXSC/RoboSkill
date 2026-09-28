#!/usr/bin/env python3
"""Schedule native Codex CLI LIBERO-40 cells with frozen filesystem priors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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
CELL = REPO_ROOT / "harness/codex/program_prior_cell.py"
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


def frame_count(workspace: str | None) -> int:
    if not workspace:
        return 0
    index = Path(workspace) / "libero-artifacts" / "frames.jsonl"
    if not index.is_file():
        return 0
    with index.open("rb") as handle:
        return sum(1 for _ in handle)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_prior(root: Path) -> None:
    validation = load_object(root / "validation.json")
    if validation.get("status") != "validated":
        raise SystemExit(f"prior is not validated: {root}")
    index = load_object(root / "program" / "program-index.json")
    modules = index.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SystemExit(f"prior has no program modules: {root}")
    for module in modules:
        filename = module.get("filename")
        expected = module.get("source_sha256")
        path = root / "program" / "modules" / str(filename)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"prior module hash mismatch: {path}")


def find_task_prior(source_root: Path, benchmark: str, task_id: int) -> tuple[Path, int]:
    """Return the single selected, validated experience for one benchmark task."""

    task_root = source_root / benchmark / f"task-{task_id:02d}"
    candidates = sorted(path for path in task_root.glob("seed-*") if path.is_dir())
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one selected prior under {task_root}, found {len(candidates)}"
        )
    source = candidates[0]
    validate_prior(source)
    try:
        source_seed = int(source.name[len("seed-"):])
    except ValueError as error:
        raise SystemExit(f"invalid prior seed directory: {source}") from error
    return source, source_seed


def make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def source_task_id(spec: dict[str, Any], benchmark: str, task_id: int) -> int:
    mapping = spec.get("source_task_map", {}).get(benchmark, {})
    return int(mapping.get(str(task_id), mapping.get(task_id, task_id)))


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
    source_root = public_path(spec["experience_source_root"])
    snapshot_root = batch_root / "program-prior-snapshot"
    state_path = batch_root / "batch-state.json"
    if args.resume_existing:
        state = load_object(state_path)
        jobs = state.get("jobs")
        if not isinstance(jobs, list):
            raise SystemExit("existing batch state has no jobs list")
    else:
        batch_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        atomic_json(batch_root / "batch-plan.json", spec)
        snapshot_manifest: dict[str, Any] = {
            "schema_version": 1,
            "source_root": str(source_root),
            "tasks": [],
        }
        source_seeds: dict[tuple[str, int], int] = {}
        for group in spec["task_groups"]:
            benchmark = str(group["benchmark"])
            for task_id in group["task_ids"]:
                task_id = int(task_id)
                mapped_task_id = source_task_id(spec, benchmark, task_id)
                source, source_seed = find_task_prior(source_root, benchmark, task_id)
                source_seeds[(benchmark, task_id)] = source_seed
                destination = snapshot_root / benchmark / f"task-{int(task_id):02d}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
                frozen_hash = tree_sha256(destination)
                make_read_only(destination)
                snapshot_manifest["tasks"].append({
                    "benchmark": benchmark,
                    "task_id": int(task_id),
                    "source_task_id": mapped_task_id,
                    "source_seed": source_seed,
                    "source": str(source),
                    "snapshot": str(destination),
                    "sha256": frozen_hash,
                })
        atomic_json(batch_root / "experience-snapshot-manifest.json", snapshot_manifest)
        jobs: list[dict[str, Any]] = []
        index = 0
        for seed in spec["seeds"]:
            for group in spec["task_groups"]:
                for task_id in group["task_ids"]:
                    mapped_task_id = source_task_id(
                        spec, str(group["benchmark"]), int(task_id)
                    )
                    index += 1
                    jobs.append(
                        {
                            "index": index,
                            "benchmark": str(group["benchmark"]),
                            "task_id": int(task_id),
                            "prior_source_task_id": mapped_task_id,
                            "seed": int(seed),
                            "prior_source_seed": source_seeds[(str(group["benchmark"]), int(task_id))],
                            "transfer_class": "cross-task" if mapped_task_id != int(task_id) else (
                                "same-seed-replay"
                                if int(seed) == source_seeds[(str(group["benchmark"]), int(task_id))]
                                else "cross-seed"
                            ),
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
            model=str(spec["model"]["name"]),
            agent_preset="codex-cli",
            workflow_variant="plain",
            continue_on_unsuccessful_turn=True,
            active_response_timeout_seconds=None,
            label=(
                f"codex-sol-prior-{benchmark}-task{int(job['task_id']):02d}-"
                + (f"from-task{int(job['prior_source_task_id']):02d}-" if int(job["prior_source_task_id"]) != int(job["task_id"]) else "")
                + f"seed{int(job['seed'])}-l4"
            ),
        )
        frozen_prior = snapshot_root / benchmark / f"task-{int(job['task_id']):02d}"
        validate_prior(frozen_prior)
        prior_hash = tree_sha256(frozen_prior)
        workspace_prior = (
            workspace / "program-prior" / benchmark / f"task-{int(job['task_id']):02d}"
        )
        workspace_prior.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(frozen_prior, workspace_prior)
        make_read_only(workspace_prior)
        config = load_object(config_path)
        cell_home = workspace / ".codex-home"
        cell_home.mkdir(mode=0o700)
        if provider_id == "openai-chatgpt-auth":
            auth_file = Path(spec["codex_auth_file"]).expanduser().resolve()
            (cell_home / "auth.json").symlink_to(auth_file)
        config["codex"] = {
            "executable": str(spec["codex_executable"]),
            "home": str(cell_home),
            "model": str(spec["model"]["name"]),
            "reasoning_effort": str(spec["model"]["reasoning_effort"]),
            "context_window": spec["model"].get("context_window"),
            "auto_compact_token_limit": spec["model"].get("auto_compact_token_limit"),
            "provider": provider,
        }
        config["program_prior"] = {
            "path": str(workspace_prior),
            "source": str(frozen_prior),
            "snapshot_sha256": prior_hash,
            "source_seed": int(job["prior_source_seed"]),
            "source_task_id": int(job["prior_source_task_id"]),
            "target_task_id": int(job["task_id"]),
            "transfer_class": str(job["transfer_class"]),
        }
        config["post_success_review"] = {"enabled": False}
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
            program_prior=str(frozen_prior),
            program_prior_sha256=prior_hash,
            prior_source_seed=int(job["prior_source_seed"]),
            prior_source_task_id=int(job["prior_source_task_id"]),
            transfer_class=str(job["transfer_class"]),
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
