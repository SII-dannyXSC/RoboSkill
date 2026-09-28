#!/usr/bin/env python3
"""Grow E0 -> E1 -> E2 for ten LIBERO-10 tasks, then run seed-major evaluation."""

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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUNTIME = REPO_ROOT / "libero/runtime/full"
sys.path.insert(0, str(RUNTIME))

from launch_attempt import prepare_attempt  # noqa: E402
from libero_runtime import atomic_json  # noqa: E402


TERMINAL = {"success", "finished-no-success", "controller-error", "experience-failed", "cancelled"}
CODEX_PLAIN_CELL = REPO_ROOT / "harness/codex/cell.py"
CODEX_PRIOR_CELL = REPO_ROOT / "harness/codex/evolution_cell.py"
CLAUDE_PLAIN_CELL = REPO_ROOT / "harness/claude/cell.py"
CLAUDE_PRIOR_CELL = REPO_ROOT / "harness/claude/program_prior_cell.py"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one JSON object")
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
    path = Path(workspace) / "libero-artifacts/frames.jsonl"
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_experience(root: Path) -> None:
    validation = load_object(root / "validation.json")
    if validation.get("status") != "validated":
        raise SystemExit(f"experience is not validated: {root}")
    index = load_object(root / "program/program-index.json")
    modules = index.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SystemExit(f"experience has no modules: {root}")
    for module in modules:
        filename = str(module.get("filename", ""))
        path = root / "program/modules" / filename
        if not path.is_file():
            raise SystemExit(f"experience module is missing: {path}")
        expected = module.get("source_sha256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"experience module hash mismatch: {path}")
    if not (root / "episode-summary.md").is_file():
        raise SystemExit(f"experience summary is missing: {root}")


def make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def public_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("model_key", choices=("codex", "opus"))
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    spec = load_object(args.config.expanduser().resolve())
    model_key = args.model_key
    model = spec["models"][model_key]
    executor = str(model["executor"])
    provider_id = str(model["provider"]["id"])
    if executor == "codex" and provider_id != "openai-chatgpt-auth":
        env_key = str(model["provider"].get("env_key", ""))
        if not env_key or not os.environ.get(env_key):
            raise SystemExit(f"missing custom-provider credential: {env_key or 'env_key'}")

    experiment_root = public_path(spec["experiment_root"])
    model_root = experiment_root / "models" / model_key
    batch_root = model_root / "batches" / str(model["batch_id"])
    state_path = batch_root / "experiment-state.json"
    experiences_root = batch_root / "experiences"
    tasks = [int(v) for v in spec["task_ids"]]
    max_concurrency = int(model["concurrency"])
    retry_limit = int(spec.get("startup_retry_limit", 3))
    retry_delay = float(spec.get("startup_retry_delay_seconds", 60))
    generation_attempts = int(spec["generation"]["max_attempts_per_stage"])
    children: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    stop = False

    def handle_stop(*_: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    if args.resume_existing:
        state = load_object(state_path)
        jobs = state["jobs"]
        state.update(status="running", resumed_at=now_iso(), scheduler_pid=os.getpid())
    else:
        batch_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        atomic_json(batch_root / "experiment-plan.json", {**spec, "active_model": model_key})
        disabled_claim = batch_root / "disabled-review-claim.json"
        atomic_json(disabled_claim, {"disabled": True, "reason": "evaluation cells never write experience"})
        experience_map: dict[str, list[dict[str, Any]]] = {}
        manifest: list[dict[str, Any]] = []
        e0_root = public_path(model["e0_root"])
        for task_id in tasks:
            source_seed = int(model["e0_source_seeds"][str(task_id)])
            source = e0_root / f"task-{task_id:02d}"
            if executor == "codex":
                source = source / f"seed-{source_seed}"
            validate_experience(source)
            destination = experiences_root / f"task-{task_id:02d}/e0"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            digest = tree_sha256(destination)
            make_read_only(destination)
            item = {
                "experience_id": "e0", "path": str(destination), "source": str(source),
                "source_seed": source_seed, "sha256": digest,
            }
            experience_map[str(task_id)] = [item]
            manifest.append({"task_id": task_id, **item})
        atomic_json(batch_root / "e0-snapshot-manifest.json", {"schema_version": 1, "items": manifest})
        jobs: list[dict[str, Any]] = []
        state = {
            "schema_version": 1,
            "model_key": model_key,
            "batch_id": model["batch_id"],
            "status": "running",
            "phase": "generation-e1",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "scheduler_pid": os.getpid(),
            "max_concurrency": max_concurrency,
            "experience_map": experience_map,
            "evaluation_seed_index": 0,
            "evaluation_created_seeds": [],
            "jobs": jobs,
        }

    def save() -> None:
        state["updated_at"] = now_iso()
        state["summary"] = dict(Counter(job["state"] for job in jobs))
        state["phase_summary"] = {
            phase: dict(Counter(j["state"] for j in jobs if j["phase"] == phase))
            for phase in sorted({str(j["phase"]) for j in jobs})
        }
        atomic_json(state_path, state)

    def experience(task_id: int, experience_id: str) -> dict[str, Any]:
        matches = [x for x in state["experience_map"][str(task_id)] if x["experience_id"] == experience_id]
        if len(matches) != 1:
            raise SystemExit(f"missing {experience_id} for task {task_id}")
        return matches[0]

    def add_generation_job(task_id: int, stage: str, attempt: int) -> None:
        parent_id = "e0" if stage == "e1" else "e1"
        seeds = [int(v) for v in spec["generation"][f"{stage}_seeds"]]
        parent = experience(task_id, parent_id)
        jobs.append({
            "index": len(jobs) + 1,
            "job_id": f"generation-{stage}-task-{task_id:02d}-attempt-{attempt + 1}",
            "phase": f"generation-{stage}", "task_id": task_id, "seed": seeds[attempt],
            "experience_id": parent_id, "prior_source_seed": int(parent["source_seed"]),
            "generation_stage": stage, "generation_attempt": attempt,
            "state": "queued", "pid": None, "workspace": None, "started_at": None,
            "finished_at": None, "outcome_reason": None, "retry_count": 0,
            "previous_attempts": [],
        })

    def create_generation_stage(stage: str) -> None:
        for task_id in tasks:
            add_generation_job(task_id, stage, 0)
        state["phase"] = f"generation-{stage}"

    def create_evaluation_wave(seed: int) -> None:
        conditions = [str(v) for v in spec["conditions"]]
        # Condition rounds cover every task before returning to the same task.
        for round_index, condition in enumerate(conditions):
            rotated = tasks[round_index:] + tasks[:round_index]
            for task_id in rotated:
                source_seed = None
                if condition != "plain":
                    source_seed = int(experience(task_id, condition)["source_seed"])
                jobs.append({
                    "index": len(jobs) + 1,
                    "job_id": f"evaluation-seed-{seed}-task-{task_id:02d}-{condition}",
                    "phase": "evaluation", "evaluation_seed": seed, "task_id": task_id,
                    "seed": seed, "experience_id": condition,
                    "prior_source_seed": source_seed, "state": "queued", "pid": None,
                    "workspace": None, "started_at": None, "finished_at": None,
                    "outcome_reason": None, "retry_count": 0, "previous_attempts": [],
                })
        state["evaluation_created_seeds"].append(seed)
        state["phase"] = f"evaluation-seed-{seed}"

    def requeue_startup_failure(job: dict[str, Any], reason: str) -> bool:
        if frame_count(job.get("workspace")) != 0:
            return False
        retry_count = int(job.get("retry_count", 0))
        if retry_count >= retry_limit:
            return False
        job["previous_attempts"].append({
            "workspace": job.get("workspace"), "pid": job.get("pid"),
            "started_at": job.get("started_at"), "finished_at": now_iso(), "reason": reason,
        })
        job.update(state="queued", pid=None, workspace=None, started_at=None, finished_at=None,
                   outcome_reason=None, retry_count=retry_count + 1,
                   retry_not_before=time.time() + retry_delay)
        return True

    def requeue_interrupted_job(job: dict[str, Any], reason: str) -> None:
        """Retry an externally interrupted cell without counting it as an evaluation result."""
        job["previous_attempts"].append({
            "workspace": job.get("workspace"), "pid": job.get("pid"),
            "started_at": job.get("started_at"), "finished_at": now_iso(),
            "reason": reason, "excluded_from_results": True,
        })
        job.update(
            state="queued", pid=None, workspace=None, started_at=None, finished_at=None,
            outcome_reason=None, retry_not_before=time.time(),
            interruption_retry_count=int(job.get("interruption_retry_count", 0)) + 1,
        )

    def launch(job: dict[str, Any]) -> None:
        task_id = int(job["task_id"])
        prior_id = str(job["experience_id"])
        has_prior = prior_id != "plain"
        attempt_id = f"{model_key}-{uuid.uuid4().hex[:7]}-j{int(job['index']):03d}"
        workspace, config_path, _ = prepare_attempt(
            experiment_root=model_root, benchmark=str(spec["benchmark"]), task_id=task_id,
            level=int(spec["observation_level"]), attempt_id=attempt_id,
            episode_length=int(spec["episode_length"]), bbox_scope=str(spec["bbox_scope"]),
            seed=int(job["seed"]), deadline_seconds=int(spec["deadline_seconds"]),
            gateway_url=str(spec["gateway_url"]), harness_url="http://127.0.0.1:1",
            provider=str(model["provider"]["id"]), model=str(model["model"]),
            agent_preset="codex-cli" if executor == "codex" else "claude-code",
            workflow_variant="plain", continue_on_unsuccessful_turn=True,
            active_response_timeout_seconds=None,
            label=f"{model_key}-l10-t{task_id:02d}-{job['phase']}-{prior_id}-s{job['seed']}",
        )
        config = load_object(config_path)
        if executor == "codex":
            cell_home = workspace / ".codex-home"
            cell_home.mkdir(mode=0o700)
            if provider_id == "openai-chatgpt-auth":
                auth_file = Path(str(model["auth_file"])).expanduser().resolve()
                (cell_home / "auth.json").symlink_to(auth_file)
            config["codex"] = {
                "executable": str(model["executable"]),
                "home": str(cell_home), "model": str(model["model"]),
                "reasoning_effort": str(model["reasoning_effort"]),
                "context_window": model.get("context_window"),
                "auto_compact_token_limit": model.get("auto_compact_token_limit"),
                "provider": model["provider"],
            }
            cell_script = CODEX_PRIOR_CELL if has_prior else CODEX_PLAIN_CELL
        else:
            config["claude_code"] = {
                "executable": str(model["executable"]), "model": str(model["model"]),
                "effort": str(model["reasoning_effort"]),
                "autocompact": str(model.get("autocompact", "auto")),
                "provider": model["provider"], "session_id": str(uuid.uuid4()),
            }
            cell_script = CLAUDE_PRIOR_CELL if has_prior else CLAUDE_PLAIN_CELL

        prior_source: Path | None = None
        prior_hash: str | None = None
        if has_prior:
            item = experience(task_id, prior_id)
            prior_source = Path(item["path"])
            validate_experience(prior_source)
            prior_hash = tree_sha256(prior_source)
            if executor == "codex":
                workspace_prior = workspace / "experience"
            else:
                workspace_prior = workspace / "program-prior/libero_10" / f"task-{task_id:02d}"
                workspace_prior.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(prior_source, workspace_prior)
            make_read_only(workspace_prior)
            config["program_prior"] = {
                "path": str(workspace_prior), "source": str(prior_source),
                "snapshot_sha256": prior_hash, "source_seed": job.get("prior_source_seed"),
                "source_task_id": task_id, "target_task_id": task_id,
                "transfer_class": "cross-seed", "experience_id": prior_id,
            }

        if job["phase"].startswith("generation-"):
            review_root = batch_root / "generation-reviews" / str(job["job_id"])
            export = review_root / "experience-reviews/libero_10" / f"task-{task_id:02d}" / f"seed-{int(job['seed'])}"
            config["post_success_review"] = {
                "enabled": True, "claim_path": str(review_root / "reviewer-claim.json"),
                "batch_root": str(review_root), "export_path": str(export),
                "turn_grace_seconds": 90, "timeout_seconds": int(spec["review_timeout_seconds"]),
                "max_attempts": 2,
            }
        elif has_prior:
            config["post_success_review"] = {"enabled": False}
        else:
            config["post_success_review"] = {
                "enabled": False,
                "claim_path": str(batch_root / "disabled-review-claim.json"),
                "export_path": str(batch_root / "disabled-review-export"),
                "turn_grace_seconds": 1, "timeout_seconds": 1, "max_attempts": 1,
            }
        atomic_json(config_path, config)
        log_path = workspace / "logs/launcher.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            [sys.executable, str(cell_script), str(config_path)], cwd=workspace,
            env={**os.environ, "LIBERO_RUNTIME_DIR": str(RUNTIME)},
            stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
            text=True, start_new_session=True,
        )
        children[process.pid] = (process, log_handle)
        job.update(state="running", pid=process.pid, workspace=str(workspace),
                   started_at=now_iso(), experience_path=str(prior_source) if prior_source else None,
                   experience_sha256=prior_hash)

    def promote(job: dict[str, Any], result: dict[str, Any]) -> bool:
        export_value = result.get("review_export_path")
        if result.get("review_status") != "validated" or not isinstance(export_value, str):
            return False
        source = Path(export_value)
        if not source.is_dir():
            return False
        validate_experience(source)
        stage = str(job["generation_stage"])
        destination = experiences_root / f"task-{int(job['task_id']):02d}" / stage
        if destination.exists():
            raise SystemExit(f"refusing to overwrite promoted experience: {destination}")
        shutil.copytree(source, destination)
        digest = tree_sha256(destination)
        make_read_only(destination)
        item = {
            "experience_id": stage, "parent_experience_id": job["experience_id"],
            "path": str(destination), "source": str(source), "source_seed": int(job["seed"]),
            "sha256": digest, "writer_job_id": job["job_id"],
        }
        state["experience_map"][str(job["task_id"])].append(item)
        job["promoted_experience_id"] = stage
        job["promoted_experience_sha256"] = digest
        return True

    def correct_plain_metadata(workspace: Path) -> None:
        path = workspace / "libero-evaluation.json"
        if not path.is_file():
            return
        value = load_object(path)
        value["provider"] = str(model["provider"]["id"])
        value["imported_experience"] = False
        value.pop("program_prior", None)
        atomic_json(path, value)

    def reap() -> None:
        for job in list(jobs):
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
            workspace = Path(str(job["workspace"]))
            result_path = workspace / ".libero-controller-result.json"
            result = load_object(result_path) if result_path.is_file() else {}
            success = result.get("success") is True
            reason = str(result.get("reason", "controller_missing_result"))
            if job["experience_id"] == "plain":
                correct_plain_metadata(workspace)
            if reason == "stopped":
                requeue_interrupted_job(job, "infrastructure_interruption_stopped")
                continue
            if reason == "controller_error" and requeue_startup_failure(job, reason):
                continue
            promoted = True
            if job["phase"].startswith("generation-"):
                promoted = success and promote(job, result)
            final_state = "success" if success and promoted else (
                "experience-failed" if success else (
                    "controller-error" if reason == "controller_error" else "finished-no-success"
                )
            )
            job.update(state=final_state, finished_at=now_iso(), outcome_reason=reason,
                       review_status=result.get("review_status", "not-selected"),
                       agent_session_id=result.get("codex_thread_id") or result.get("claude_session_id"))
            if job["phase"].startswith("generation-") and final_state != "success":
                attempt = int(job["generation_attempt"])
                if attempt + 1 < generation_attempts:
                    add_generation_job(int(job["task_id"]), str(job["generation_stage"]), attempt + 1)

    if not args.resume_existing:
        create_generation_stage("e1")
    save()
    try:
        while not stop:
            reap()
            phase = str(state["phase"])
            if phase == "generation-e1" and all(
                any(x["experience_id"] == "e1" for x in state["experience_map"][str(t)]) for t in tasks
            ):
                create_generation_stage("e2")
                phase = str(state["phase"])
            if phase == "generation-e2" and all(
                any(x["experience_id"] == "e2" for x in state["experience_map"][str(t)]) for t in tasks
            ):
                create_evaluation_wave(int(spec["evaluation_seeds"][0]))
                phase = str(state["phase"])

            generation_phase = phase.startswith("generation-")
            if generation_phase:
                stage = phase[len("generation-"):]
                exhausted = []
                for task_id in tasks:
                    if any(x["experience_id"] == stage for x in state["experience_map"][str(task_id)]):
                        continue
                    task_jobs = [j for j in jobs if j.get("generation_stage") == stage and int(j["task_id"]) == task_id]
                    if len(task_jobs) >= generation_attempts and all(j["state"] in TERMINAL for j in task_jobs):
                        exhausted.append(task_id)
                if exhausted:
                    state.update(status="blocked-experience-generation", blocked_tasks=exhausted)
                    save()
                    return 2

            if phase.startswith("evaluation-seed-"):
                current_seed = int(phase.rsplit("-", 1)[1])
                wave = [j for j in jobs if j["phase"] == "evaluation" and int(j["evaluation_seed"]) == current_seed]
                if wave and all(j["state"] in TERMINAL for j in wave):
                    seeds = [int(v) for v in spec["evaluation_seeds"]]
                    position = seeds.index(current_seed)
                    if position + 1 == len(seeds):
                        state["status"] = "completed"
                        state["phase"] = "completed"
                        save()
                        return 0
                    create_evaluation_wave(seeds[position + 1])
                    phase = str(state["phase"])

            running = sum(j["state"] == "running" for j in jobs)
            eligible = [
                j for j in jobs if j["state"] == "queued"
                and float(j.get("retry_not_before", 0)) <= time.time()
                and (
                    (phase.startswith("generation-") and j["phase"] == phase)
                    or (phase.startswith("evaluation-seed-") and j["phase"] == "evaluation"
                        and int(j["evaluation_seed"]) == int(phase.rsplit("-", 1)[1]))
                )
            ]
            for job in eligible[: max(0, max_concurrency - running)]:
                launch(job)
            save()
            time.sleep(2)
    finally:
        if stop:
            state["status"] = "stopping"
            save()
            for job in jobs:
                if job["state"] == "running" and job.get("pid"):
                    try:
                        os.killpg(int(job["pid"]), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and any(
                j["state"] == "running" and process_alive(int(j["pid"])) for j in jobs
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
