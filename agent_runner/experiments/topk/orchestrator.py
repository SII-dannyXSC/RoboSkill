#!/usr/bin/env python3
"""Evaluate all LIBERO-10 tasks with an agent-selected Top-K experience catalog."""

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
sys.path.insert(0, str(HERE))

from launch_attempt import prepare_attempt  # noqa: E402
from libero_runtime import atomic_json  # noqa: E402
from retriever import Result, normalize_tasks, retrieve  # noqa: E402

CELL = REPO_ROOT / "harness/library_prior_cell.py"
INFRA_PATTERNS = (
    "stream disconnected before completion", "error sending request", "connection reset",
    "connection refused", "timed out", "overloaded_error", "rate_limit_error",
)
TERMINAL = {"success", "clean-failure", "infra-failed", "controller-error", "cancelled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_experience(root: Path) -> None:
    if load_object(root / "validation.json").get("status") != "validated":
        raise ValueError(f"experience is not validated: {root}")
    modules = load_object(root / "program/program-index.json").get("modules")
    if not isinstance(modules, list) or not modules:
        raise ValueError(f"experience has no indexed modules: {root}")
    for item in modules:
        module = root / "program/modules" / str(item["filename"])
        if not module.is_file():
            raise ValueError(f"missing module: {module}")
        expected = item.get("source_sha256")
        if expected and hashlib.sha256(module.read_bytes()).hexdigest() != expected:
            raise ValueError(f"module hash mismatch: {module}")


def make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def public_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def frame_stats(workspace: Path) -> dict[str, Any]:
    path = workspace / "libero-artifacts/frames.jsonl"
    frames = steps = 0
    episodes: set[int] = set()
    first = last = None
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                frames += 1
                if isinstance(row.get("step_index"), int) and row["step_index"] > 0:
                    steps += 1
                if isinstance(row.get("episode_index"), int):
                    episodes.add(row["episode_index"])
                stamp = row.get("timestamp") or row.get("created_at")
                if isinstance(stamp, str):
                    first = first or stamp
                    last = stamp
    return {"frames": frames, "steps": steps, "episodes": len(episodes), "first_frame_at": first, "last_frame_at": last}


def infra_summary(workspace: Path) -> dict[str, Any]:
    counts = Counter()
    for pattern in ("*stream.jsonl", "*stderr.log"):
        for path in (workspace / "logs").glob(pattern):
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for needle in INFRA_PATTERNS:
                counts[needle] += text.count(needle)
    return {"contaminated": bool(counts), "pattern_counts": dict(counts)}


def infer_selection(workspace: Path, catalog: Path) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, str]]] = {}
    by_name: dict[str, list[dict[str, str]]] = {}
    for candidate in load_object(catalog / "program/program-index.json")["candidates"]:
        experience_id = str(candidate["experience_id"])
        for module in (catalog / candidate["relative_path"] / "program/modules").iterdir():
            if not module.is_file():
                continue
            source = {"experience_id": experience_id, "filename": module.name}
            by_hash.setdefault(hashlib.sha256(module.read_bytes()).hexdigest(), []).append(source)
            by_name.setdefault(module.name, []).append(source)
    exact, modified = [], []
    agent = workspace / "agent"
    if agent.is_dir():
        for path in sorted(p for p in agent.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            row = {"agent_path": path.relative_to(agent).as_posix()}
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in by_hash:
                exact.append({**row, "source_matches": by_hash[digest]})
            elif path.name in by_name:
                modified.append({**row, "possible_sources": by_name[path.name]})
    return {"schema_version": 1, "method": "exact hash plus filename hint", "exact_copies": exact,
            "possible_modified_copies": modified, "note": "No match does not prove non-use."}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("model_key")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--retry-cell",
        action="append",
        default=[],
        metavar="TASK_ID:SEED",
        help="append one new attempt for an exhausted cell while resuming",
    )
    args = parser.parse_args()
    spec = load_object(args.config.resolve())
    if args.model_key not in spec["models"]:
        raise SystemExit(f"unknown model key: {args.model_key}")
    model_key, model = args.model_key, spec["models"][args.model_key]
    executor = str(model["executor"])
    provider_id = str(model["provider"]["id"])
    if executor == "codex" and provider_id != "openai-chatgpt-auth":
        env_key = str(model["provider"].get("env_key", ""))
        if not env_key or not os.environ.get(env_key):
            raise SystemExit(f"missing custom-provider credential: {env_key or 'env_key'}")

    root = public_path(spec["experiment_root"])
    batch = root / "models" / model_key / "batches" / str(model["batch_id"])
    state_path = batch / "experiment-state.json"
    tasks = normalize_tasks(spec["tasks"])
    evaluation_seeds = [int(seed) for seed in spec["evaluation_seeds"]]
    max_concurrency = int(model["concurrency"])
    retry_cells: set[tuple[int, int]] = set()
    for value in args.retry_cell:
        try:
            task_text, seed_text = value.split(":", 1)
            cell = (int(task_text), int(seed_text))
        except (TypeError, ValueError) as error:
            raise SystemExit(f"invalid --retry-cell {value!r}; expected TASK_ID:SEED") from error
        if cell[0] not in range(10) or cell[1] not in evaluation_seeds:
            raise SystemExit(f"--retry-cell is outside the configured matrix: {value!r}")
        retry_cells.add(cell)
    if retry_cells and not args.resume_existing:
        raise SystemExit("--retry-cell requires --resume-existing")
    if args.resume_existing:
        state = load_object(state_path)
        if any(job["state"] == "running" for job in state.get("jobs", [])):
            raise SystemExit("cannot resume a state containing running jobs")
        configured_cells = {(task_id, seed) for task_id in range(10) for seed in evaluation_seeds}
        jobs = [job for job in state["jobs"]
                if (int(job["task_id"]), int(job["seed"])) in configured_cells]
        state["jobs"] = jobs
        existing_cells = {(int(job["task_id"]), int(job["seed"])) for job in jobs}
        for task_id in range(10):
            for seed in evaluation_seeds:
                if (task_id, seed) in existing_cells:
                    continue
                jobs.append({"index": len(jobs) + 1,
                    "job_id": f"task-{task_id:02d}-seed-{seed}-attempt-1",
                    "task_id": task_id, "seed": seed, "attempt": 1, "state": "queued",
                    "pid": None, "workspace": None, "started_at": None, "finished_at": None,
                    "outcome_reason": None})
        for task_id, seed in sorted(retry_cells):
            prior_attempts = [job for job in jobs
                              if int(job["task_id"]) == task_id and int(job["seed"]) == seed]
            if not prior_attempts:
                raise SystemExit(f"cannot retry missing cell {task_id}:{seed}")
            if any(job["state"] == "running" for job in prior_attempts):
                raise SystemExit(f"cannot retry running cell {task_id}:{seed}")
            previous = max(prior_attempts, key=lambda job: int(job["attempt"]))
            attempt = int(previous["attempt"]) + 1
            jobs.append({"index": len(jobs) + 1,
                "job_id": f"task-{task_id:02d}-seed-{seed}-attempt-{attempt}",
                "task_id": task_id, "seed": seed, "attempt": attempt, "state": "queued",
                "pid": None, "workspace": None, "started_at": None, "finished_at": None,
                "outcome_reason": None, "retry_of": previous["job_id"],
                "manual_retry": True})
        state.update(batch_id=model["batch_id"], scheduler_pid=os.getpid(), max_concurrency=max_concurrency,
                     evaluation_seeds=evaluation_seeds, resumed_at=now_iso())
    else:
        batch.mkdir(parents=True, exist_ok=False, mode=0o700)
        jobs = [{"index": index + 1, "job_id": f"task-{task_id:02d}-seed-{seed}-attempt-1",
                 "task_id": task_id, "seed": seed, "attempt": 1, "state": "queued",
                 "pid": None, "workspace": None, "started_at": None, "finished_at": None,
                 "outcome_reason": None}
                for index, (task_id, seed) in enumerate(
                    (task_id, seed) for task_id in range(10) for seed in evaluation_seeds)]
        state = {"schema_version": 1, "experiment_id": spec["experiment_id"],
            "model_key": model_key, "batch_id": model["batch_id"], "status": "waiting-prerequisite",
            "scheduler_pid": os.getpid(), "max_concurrency": max_concurrency,
            "evaluation_seeds": evaluation_seeds,
            "created_at": now_iso(), "updated_at": now_iso(),
            "prerequisite_state": model.get("prerequisite_state"), "jobs": jobs}
    children: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    stopping = False

    def stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def save() -> None:
        state["updated_at"] = now_iso()
        state["summary"] = dict(Counter(job["state"] for job in jobs))
        atomic_json(state_path, state)

    def prerequisite_done() -> bool:
        configured = spec.get("prerequisite_states")
        paths = (
            [public_path(str(value)) for value in configured]
            if isinstance(configured, list)
            else [public_path(str(model["prerequisite_state"]))]
        )
        observed: dict[str, Any] = {}
        complete = True
        for path in paths:
            if not path.is_file():
                observed[str(path)] = {"status": "missing"}; complete = False; continue
            prior = load_object(path)
            observed[str(path)] = {"status": prior.get("status"), "updated_at": prior.get("updated_at")}
            if prior.get("status") != "completed": complete = False
        state["prerequisite_observed"] = observed
        state["prerequisite_observed_status"] = "completed" if complete else "waiting"
        return complete

    def build_catalogs() -> None:
        source_root = public_path(spec["source_experience_root"]) / str(model["source_directory"])
        source_hashes: dict[str, str] = {}
        for item in tasks:
            source = source_root / f"task-{int(item['task_id']):02d}"
            validate_experience(source)
            source_hashes[str(item["task_id"])] = tree_sha256(source)
        manifest: dict[str, Any] = {"schema_version": 1, "retriever": "metadata-topk-agent-selector",
            "top_k": spec["retrieval"]["top_k"], "threshold": spec["retrieval"]["threshold"], "tasks": {}}
        for query in tasks:
            fixed = spec.get("fixed_selections", {}).get(model_key, {}).get(str(query["task_id"]))
            if fixed is not None:
                selected = [Result(experience_id=str(value), score=round(1.0-rank*0.01, 6), recall=0.0,
                    jaccard=0.0, keyword_bonus=0.0, overlap=[]) for rank, value in enumerate(fixed)]
                ranked = selected
                manifest["retriever"] = "frozen-agent-selected-catalog"
            else:
                selected, ranked = retrieve(query, tasks, int(spec["retrieval"]["top_k"]), float(spec["retrieval"]["threshold"]))
            catalog = batch / "catalogs" / f"task-{int(query['task_id']):02d}"
            entries = []
            for result in selected:
                source_id = int(result.experience_id)
                source = source_root / f"task-{source_id:02d}"
                destination = catalog / "candidates" / f"task-{source_id:02d}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, destination)
                entries.append({"experience_id": result.experience_id, "task": spec["tasks"][str(source_id)],
                    "relative_path": f"candidates/task-{source_id:02d}",
                    "program_index": f"candidates/task-{source_id:02d}/program/program-index.json",
                    "snapshot_sha256": tree_sha256(destination), "retrieval_score": result.score})
            (catalog / "program").mkdir(parents=True)
            atomic_json(catalog / "program/program-index.json", {"schema_version": 1,
                "summary": "Independent validated experiences selected from the LIBERO-10 library.",
                "selection_policy": "Agent selects modules per current observation; candidates remain immutable.",
                "target_task_id": query["task_id"], "candidates": entries})
            atomic_json(catalog / "validation.json", {"schema_version": 1,
                "status": "validated-candidate-catalog", "candidate_count": len(entries)})
            manifest["tasks"][str(query["task_id"])] = {"selected": [x.to_dict() for x in selected],
                "ranked": [x.to_dict() for x in ranked], "catalog_sha256": tree_sha256(catalog)}
            make_read_only(catalog)
        manifest["source_hashes"] = source_hashes
        atomic_json(batch / "retrieval-manifest.json", manifest)
        atomic_json(batch / "experiment-plan.json", {**spec, "active_model": model_key})
        state["retrieval_manifest"] = str(batch / "retrieval-manifest.json")

    def add_retry(job: dict[str, Any]) -> None:
        attempt = int(job["attempt"]) + 1
        jobs.append({"index": len(jobs) + 1, "job_id": f"task-{job['task_id']:02d}-seed-{job['seed']}-attempt-{attempt}",
            "task_id": int(job["task_id"]), "seed": int(job["seed"]), "attempt": attempt, "state": "queued",
            "pid": None, "workspace": None, "started_at": None, "finished_at": None, "outcome_reason": None,
            "retry_of": job["job_id"]})

    def launch(job: dict[str, Any]) -> None:
        task_id = int(job["task_id"])
        attempt_id = f"{model_key}-t{task_id:02d}-s{job['seed']}-a{job['attempt']}-{uuid.uuid4().hex[:6]}"
        workspace, config_path, _ = prepare_attempt(experiment_root=root / "models" / model_key,
            benchmark=str(spec["benchmark"]), task_id=task_id, level=int(spec["observation_level"]),
            attempt_id=attempt_id, episode_length=int(spec["episode_length"]), bbox_scope=str(spec["bbox_scope"]),
            seed=int(job["seed"]), deadline_seconds=int(spec["deadline_seconds"]), gateway_url=str(spec["gateway_url"]),
            harness_url="http://127.0.0.1:1", provider=str(model["provider"]["id"]), model=str(model["model"]),
            agent_preset="codex-cli" if executor == "codex" else "claude-code", workflow_variant="plain",
            continue_on_unsuccessful_turn=True, active_response_timeout_seconds=None, label=job["job_id"])
        config = load_object(config_path)
        if executor == "codex":
            home = workspace / ".codex-home"; home.mkdir(mode=0o700)
            if provider_id == "openai-chatgpt-auth":
                auth_file = Path(str(model["auth_file"])).expanduser().resolve()
                (home / "auth.json").symlink_to(auth_file)
            config["codex"] = {"executable": model["executable"], "home": str(home), "model": model["model"],
                "reasoning_effort": model["reasoning_effort"],
                "context_window": model.get("context_window"),
                "auto_compact_token_limit": model.get("auto_compact_token_limit"),
                "provider": model["provider"]}
            prior = workspace / "experience"
        else:
            config["claude_code"] = {"executable": model["executable"], "model": model["model"],
                "effort": model["reasoning_effort"], "autocompact": model.get("autocompact", "auto"),
                "provider": model["provider"], "session_id": str(uuid.uuid4())}
            prior = workspace / "program-prior" / str(spec["benchmark"]) / f"task-{task_id:02d}"
            prior.parent.mkdir(parents=True, exist_ok=True)
        source_catalog = batch / "catalogs" / f"task-{task_id:02d}"
        shutil.copytree(source_catalog, prior)
        make_read_only(prior)
        index = load_object(source_catalog / "program/program-index.json")
        config["program_prior"] = {"path": str(prior), "source": str(source_catalog),
            "snapshot_sha256": tree_sha256(source_catalog), "source_seed": -1, "source_task_id": task_id,
            "source_benchmark": spec["benchmark"], "target_task_id": task_id, "target_benchmark": spec["benchmark"],
            "transfer_class": "libero10-metadata-topk-agent-selector", "experience_id": f"catalog-task-{task_id:02d}",
            "candidate_experience_ids": [str(x["experience_id"]) for x in index["candidates"]]}
        config["post_success_review"] = {"enabled": False}
        config["success_budget_seconds"] = int(spec["success_budget_seconds"])
        atomic_json(config_path, config)
        log = workspace / "logs/launcher.log"; log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("a", encoding="utf-8")
        process = subprocess.Popen([sys.executable, str(CELL), executor, str(config_path)], cwd=workspace,
            env={**os.environ, "LIBERO_RUNTIME_DIR": str(RUNTIME)},
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        children[process.pid] = (process, handle)
        job.update(state="running", pid=process.pid, workspace=str(workspace), started_at=now_iso(),
            candidate_experience_ids=config["program_prior"]["candidate_experience_ids"],
            catalog_sha256=config["program_prior"]["snapshot_sha256"])

    def reap() -> None:
        for job in jobs:
            if job["state"] != "running": continue
            pid = int(job["pid"]); child = children.get(pid)
            if child is not None:
                if child[0].poll() is None: continue
                child[1].close(); children.pop(pid, None)
            elif process_alive(pid): continue
            workspace = Path(str(job["workspace"])); result_path = workspace / ".libero-controller-result.json"
            result = load_object(result_path) if result_path.is_file() else {}
            infra = infra_summary(workspace); reason = str(result.get("reason", "controller_missing_result"))
            if result.get("success") is True: final = "success"
            # A controller deadline is a valid task outcome.  Log text may contain
            # unrelated "timed out" messages, which must not turn it into infra.
            elif reason == "deadline": final = "clean-failure"
            elif infra["contaminated"]: final = "infra-failed"
            elif reason in {"controller_error", "controller_missing_result"}: final = "controller-error"
            else: final = "clean-failure"
            provenance = infer_selection(workspace, batch / "catalogs" / f"task-{job['task_id']:02d}")
            atomic_json(workspace / "selection-provenance.json", provenance)
            job.update(state=final, finished_at=now_iso(), outcome_reason=reason, infra_errors=infra,
                selection_provenance=str(workspace / "selection-provenance.json"), **frame_stats(workspace),
                agent_session_id=result.get("codex_thread_id") or result.get("claude_session_id"))
            if final in {"infra-failed", "controller-error"} and int(job["attempt"]) < 3: add_retry(job)

    save()
    try:
        while not stopping and bool(spec.get("wait_for_prerequisite", True)) and not prerequisite_done():
            save(); time.sleep(30)
        if stopping: return 0
        if not bool(spec.get("wait_for_prerequisite", True)):
            state["prerequisite_observed_status"] = "bypassed-by-config"
        manifest_path = batch / "retrieval-manifest.json"
        if args.resume_existing and manifest_path.is_file():
            state["retrieval_manifest"] = str(manifest_path)
        else:
            state["status"] = "preparing-catalogs"; save(); build_catalogs()
        state["status"] = "running"
        state.setdefault("started_at", now_iso())
        save()
        target_cells = {(task_id, seed) for task_id in range(10) for seed in evaluation_seeds}
        while not stopping:
            reap()
            completed = {(int(j["task_id"]), int(j["seed"])) for j in jobs
                         if j["state"] in {"success", "clean-failure"}}
            latest_by_cell = {
                cell: max(
                    (j for j in jobs if (int(j["task_id"]), int(j["seed"])) == cell),
                    key=lambda j: int(j["attempt"]),
                )
                for cell in target_cells
            }
            exhausted = {cell for cell, job in latest_by_cell.items()
                         if int(job["attempt"]) >= 3 and job["state"] in TERMINAL}
            done = completed | exhausted
            if done == target_cells:
                state["status"] = "completed"; state["completed_at"] = now_iso(); save(); return 0
            running_count = sum(j["state"] == "running" for j in jobs)
            available_slots = max(0, max_concurrency - running_count)
            queued = [j for j in jobs if j["state"] == "queued" and
                      (int(j["task_id"]), int(j["seed"])) not in done]
            for job in queued[:available_slots]:
                launch(job)
            save(); time.sleep(2)
    finally:
        if stopping:
            for job in jobs:
                if job["state"] == "running" and job.get("pid"):
                    try: os.killpg(int(job["pid"]), signal.SIGTERM)
                    except ProcessLookupError: pass
                    job.update(state="cancelled", finished_at=now_iso(), outcome_reason="stopped")
            state["status"] = "stopped"; save()
        for process, handle in children.values():
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
