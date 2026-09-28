from __future__ import annotations

import copy
import hashlib
import json
import logging
import multiprocessing as mp
import os
import secrets
import signal
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backend import (
    BENCHMARK_TASK_COUNTS,
    EpisodeOptions,
    InvalidSessionState,
    create_episode,
)
from .evaluations import EvaluationStore, StoreError, utc_iso_from_ns
from .phases import phases_for_task
from .settings import Settings

LOGGER = logging.getLogger(__name__)


class GatewayError(Exception):
    code = "GATEWAY_ERROR"
    http_status = 500


class CapacityError(GatewayError):
    code = "GLOBAL_CAPACITY_EXHAUSTED"
    http_status = 503


class QuotaError(GatewayError):
    code = "AGENT_SESSION_QUOTA_EXCEEDED"
    http_status = 429


class NotFoundError(GatewayError):
    code = "SESSION_NOT_FOUND"
    http_status = 404


class ConflictError(GatewayError):
    code = "INVALID_SESSION_STATE"
    http_status = 409


class SessionBusy(GatewayError):
    code = "SESSION_BUSY"
    http_status = 409


class SessionNotReady(GatewayError):
    code = "SESSION_NOT_READY"
    http_status = 409


class IdempotencyConflict(GatewayError):
    code = "IDEMPOTENCY_KEY_REUSED"
    http_status = 409


class StepIndexMismatch(GatewayError):
    code = "STEP_INDEX_MISMATCH"
    http_status = 409


class RunNotFound(GatewayError):
    code = "RUN_NOT_FOUND"
    http_status = 404


class ActiveRunExists(GatewayError):
    code = "ACTIVE_RUN_EXISTS"
    http_status = 409


class RunNotActive(GatewayError):
    code = "RUN_NOT_ACTIVE"
    http_status = 409


class RunHasActiveSessions(GatewayError):
    code = "RUN_HAS_ACTIVE_SESSIONS"
    http_status = 409


class RunAlreadySucceeded(GatewayError):
    code = "RUN_ALREADY_SUCCEEDED"
    http_status = 409


class RunConfigMismatch(GatewayError):
    code = "RUN_CONFIG_MISMATCH"
    http_status = 409


class RunAttemptLimit(GatewayError):
    code = "RUN_ATTEMPT_LIMIT_REACHED"
    http_status = 409


class EvaluationRunRequired(GatewayError):
    code = "EVALUATION_RUN_REQUIRED"
    http_status = 409


class EvaluationPersistenceFailure(GatewayError):
    code = "EVALUATION_PERSISTENCE_FAILED"
    http_status = 500


class WorkerFailure(GatewayError):
    code = "SIMULATION_ERROR"
    http_status = 500


class WorkerDied(GatewayError):
    code = "SIMULATION_WORKER_DIED"
    http_status = 502


class WorkerStartFailure(GatewayError):
    code = "SIMULATION_START_FAILED"
    http_status = 500


class WorkerStartTimeout(GatewayError):
    code = "SIMULATION_START_TIMEOUT"
    http_status = 504


class WorkerTimeout(GatewayError):
    code = "SIMULATION_TIMEOUT"
    http_status = 504


class _StartCancelled(Exception):
    pass


@dataclass(frozen=True)
class SessionSpec:
    benchmark: str
    task_id: int
    seed: int
    episode_length: int
    observation_level: int
    bbox_scope: str


@dataclass
class GpuSlot:
    slot_id: int
    gpu_id: int
    assigned_session_id: Optional[str] = None
    generation: int = 0
    quarantined: bool = False


@dataclass
class SessionRecord:
    session_id: str
    owner: str
    spec: SessionSpec
    slot_id: int
    slot_generation: int
    state: str
    run_id: Optional[str]
    created_at: float
    created_at_wall: float
    last_access: float
    idempotency_key: str
    request_fingerprint: str
    process: Optional[Any] = None
    connection: Optional[Any] = None
    startup_thread: Optional[threading.Thread] = None
    startup_done: threading.Event = field(default_factory=threading.Event)
    cancel_requested: bool = False
    instruction: str = ""
    ready_at_wall: Optional[float] = None
    startup_seconds: Optional[float] = None
    completed_at_wall: Optional[float] = None
    time_to_success_seconds: Optional[float] = None
    completion_recorded: bool = False
    steps: int = 0
    episode_index: int = 0
    failure_code: Optional[str] = None
    reset_idempotency_key: Optional[str] = None
    reset_response: Optional[Dict[str, Any]] = None
    step_idempotency_key: Optional[str] = None
    step_request_fingerprint: Optional[str] = None
    step_response: Optional[Dict[str, Any]] = None
    phase_successes: List[Dict[str, Any]] = field(default_factory=list)
    operation_lock: threading.Lock = field(default_factory=threading.Lock)


def _episode_process_main(
    connection: Any,
    settings: Settings,
    gpu_id: int,
    spec: SessionSpec,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    os.environ.setdefault("MUJOCO_GL", "egl")
    episode = None
    try:
        episode = create_episode(
            settings,
            benchmark=spec.benchmark,
            task_id=spec.task_id,
            seed=spec.seed,
            gpu_id=gpu_id,
            options=EpisodeOptions(
                episode_length=spec.episode_length,
                observation_level=spec.observation_level,
                bbox_scope=spec.bbox_scope,
            ),
        )
        connection.send(
            {"kind": "ready", "ok": True, "instruction": episode.instruction}
        )
        while True:
            command = connection.recv()
            request_id = command["request_id"]
            operation = command["operation"]
            should_exit = False
            try:
                if operation == "reset":
                    result = episode.reset()
                elif operation == "step":
                    result = episode.step(tuple(command["action"]))
                elif operation == "result":
                    result = episode.result()
                elif operation == "close":
                    episode.close()
                    result = {"status": "closed"}
                    should_exit = True
                else:
                    raise ValueError("unknown worker operation")
                success_monotonic_ns = None
                phase_successes = None
                if isinstance(result, dict):
                    success_monotonic_ns = result.pop(
                        "_success_monotonic_ns", None
                    )
                    phase_successes = result.pop("_phase_successes", None)
                connection.send(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        "ok": True,
                        "result": result,
                        "success_monotonic_ns": success_monotonic_ns,
                        "phase_successes": phase_successes,
                    }
                )
            except InvalidSessionState:
                connection.send(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        "ok": False,
                        "error_code": "INVALID_SESSION_STATE",
                    }
                )
            except Exception:
                LOGGER.error(
                    "dynamic worker operation=%s failed\n%s",
                    operation,
                    traceback.format_exc(),
                )
                connection.send(
                    {
                        "kind": "response",
                        "request_id": request_id,
                        "ok": False,
                        "error_code": "SIMULATION_ERROR",
                    }
                )
                should_exit = True
            if should_exit:
                break
    except EOFError:
        pass
    except Exception:
        LOGGER.error("dynamic worker initialization failed\n%s", traceback.format_exc())
        try:
            connection.send(
                {"kind": "ready", "ok": False, "error_code": "SIMULATION_ERROR"}
            )
        except Exception:
            pass
    finally:
        if episode is not None:
            try:
                episode.close()
            except Exception:
                LOGGER.exception("failed to close dynamic episode")
        try:
            connection.close()
        except Exception:
            pass


class AuditLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: str, **fields: Any) -> None:
        payload = {"ts": time.time(), "event": event, **fields}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        except OSError:
            LOGGER.exception("failed to write audit event")


class SimulationManager:
    """Admission controller for one dynamically-created process per Session."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._context = mp.get_context("spawn")
        self._lock = threading.RLock()
        self._sessions: Dict[str, SessionRecord] = {}
        self._idempotency: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._evaluation_store = EvaluationStore(settings.evaluations_db)
        self._run_started_monotonic_ns: Dict[str, int] = {}
        self._stopping = threading.Event()
        self._audit = AuditLogger(settings.audit_log)
        self._slots: List[GpuSlot] = []
        slot_id = 0
        for gpu_id in settings.gpu_ids:
            for _ in range(settings.max_sessions_per_gpu):
                self._slots.append(GpuSlot(slot_id=slot_id, gpu_id=gpu_id))
                slot_id += 1
        self._maintenance = threading.Thread(
            target=self._maintenance_loop,
            name="libero-session-maintenance",
            daemon=True,
        )
        self._maintenance.start()

    def _allocate_slot_locked(self, session_id: str) -> GpuSlot:
        loads: Dict[int, int] = {gpu: 0 for gpu in self.settings.gpu_ids}
        for slot in self._slots:
            if slot.assigned_session_id is not None:
                loads[slot.gpu_id] += 1
        candidates = [
            slot
            for slot in self._slots
            if slot.assigned_session_id is None and not slot.quarantined
        ]
        if not candidates:
            raise CapacityError()
        slot = min(
            candidates,
            key=lambda item: (loads[item.gpu_id], item.gpu_id, item.slot_id),
        )
        slot.generation += 1
        slot.assigned_session_id = session_id
        return slot

    def _release_slot_locked(self, record: SessionRecord, *, confirmed_dead: bool) -> None:
        slot = self._slots[record.slot_id]
        if (
            slot.assigned_session_id != record.session_id
            or slot.generation != record.slot_generation
        ):
            return
        if confirmed_dead:
            slot.assigned_session_id = None
            slot.quarantined = False
        else:
            slot.quarantined = True

    @staticmethod
    def _stop_process(process: Optional[Any]) -> bool:
        if process is None:
            return True
        try:
            process.join(timeout=0.2)
        except (AssertionError, ValueError):
            return True
        if process.is_alive():
            process.terminate()
            process.join(timeout=3)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=2)
        return not process.is_alive()

    def _cleanup_record_resources(self, record: SessionRecord) -> bool:
        confirmed_dead = self._stop_process(record.process)
        if record.connection is not None:
            try:
                record.connection.close()
            except Exception:
                pass
        with self._lock:
            self._release_slot_locked(record, confirmed_dead=confirmed_dead)
        return confirmed_dead

    def _wait_message(
        self,
        record: SessionRecord,
        timeout: float,
        timeout_error: GatewayError,
        *,
        cancellable: bool = False,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancellable and record.cancel_requested:
                raise _StartCancelled()
            remaining = max(0.0, deadline - time.monotonic())
            try:
                if record.connection is not None and record.connection.poll(
                    min(0.1, remaining)
                ):
                    return record.connection.recv()
            except (EOFError, OSError, BrokenPipeError):
                if cancellable and record.cancel_requested:
                    raise _StartCancelled()
                raise WorkerDied()
            if record.process is not None and not record.process.is_alive():
                if cancellable and record.cancel_requested:
                    raise _StartCancelled()
                raise WorkerDied()
        raise timeout_error

    @staticmethod
    def _raise_store_error(exc: StoreError) -> None:
        mapping = {
            "idempotency_conflict": IdempotencyConflict,
            "active_run_exists": ActiveRunExists,
            "run_not_found": RunNotFound,
            "run_not_active": RunNotActive,
            "run_has_active_sessions": RunHasActiveSessions,
            "run_already_succeeded": RunAlreadySucceeded,
            "run_config_mismatch": RunConfigMismatch,
            "run_attempt_limit": RunAttemptLimit,
            "run_id_required": EvaluationRunRequired,
        }
        raise mapping.get(exc.reason, EvaluationPersistenceFailure)()

    def begin_run(
        self,
        owner: str,
        *,
        launcher_owner: Optional[str] = None,
        idempotency_key: str,
        request_fingerprint: str,
        label: Optional[str] = None,
        config: Dict[str, Any],
        max_attempts: int,
    ) -> Dict[str, Any]:
        run_id = "run_" + secrets.token_urlsafe(24)
        started_monotonic_ns = time.monotonic_ns()
        with self._lock:
            if any(
                record.owner == owner
                and record.state not in {"failed", "cancelled"}
                for record in self._sessions.values()
            ):
                raise RunHasActiveSessions()
            try:
                description, created = self._evaluation_store.create_run(
                    run_id=run_id,
                    owner=owner,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    label=label,
                    config=config,
                    max_attempts=max_attempts,
                    started_wall_ns=time.time_ns(),
                    launcher_owner=launcher_owner,
                )
            except StoreError as exc:
                self._raise_store_error(exc)
            except Exception as exc:
                LOGGER.exception("failed to persist evaluation run start")
                raise EvaluationPersistenceFailure() from exc
        if created:
            self._run_started_monotonic_ns[description["run_id"]] = (
                started_monotonic_ns
            )
            self._audit.write(
                "evaluation_run_started",
                owner=owner,
                run_id=description["run_id"],
                label=label,
                started_at=description["started_at"],
                config=config,
                max_attempts=max_attempts,
            )
        return description

    def run_status(self, run_id: str, launcher_owner: str) -> Dict[str, Any]:
        started = self._run_started_monotonic_ns.get(run_id)
        elapsed = None if started is None else max(0, time.monotonic_ns() - started)
        try:
            return self._evaluation_store.get_run_for_launcher(
                run_id, launcher_owner, elapsed_ns=elapsed
            )
        except StoreError as exc:
            self._raise_store_error(exc)
        except Exception as exc:
            LOGGER.exception("failed to read evaluation run")
            raise EvaluationPersistenceFailure() from exc

    def finish_run(self, run_id: str, launcher_owner: str) -> Dict[str, Any]:
        current = self.run_status(run_id, launcher_owner)
        if current["state"] != "running":
            return current
        # The trusted launcher owns Run finalization. Finishing a Run also
        # reclaims any Session the agent forgot to close.
        with self._lock:
            agent_owner = current["agent_identity"]
            session_ids = [
                record.session_id
                for record in self._sessions.values()
                if record.owner == agent_owner and record.run_id == run_id
            ]
        for session_id in session_ids:
            try:
                self.close_session(session_id, agent_owner)
            except NotFoundError:
                pass
            except (SessionBusy, EvaluationPersistenceFailure) as exc:
                raise RunHasActiveSessions() from exc

        started = self._run_started_monotonic_ns.get(run_id)
        if started is None:
            raise EvaluationPersistenceFailure()
        try:
            description = self._evaluation_store.finish_run(
                run_id,
                launcher_owner,
                elapsed_ns=max(0, time.monotonic_ns() - started),
                finished_wall_ns=time.time_ns(),
            )
        except StoreError as exc:
            self._raise_store_error(exc)
        except Exception as exc:
            LOGGER.exception("failed to finish evaluation run")
            raise EvaluationPersistenceFailure() from exc
        self._run_started_monotonic_ns.pop(run_id, None)
        self._audit.write(
            "evaluation_run_finished",
            owner=agent_owner,
            launcher_owner=launcher_owner,
            run_id=run_id,
            elapsed_seconds=description["elapsed_seconds"],
            time_to_first_success_seconds=description[
                "time_to_first_success_seconds"
            ],
            sessions_created=description["sessions_created"],
            episodes_completed=description["episodes_completed"],
            successful_episodes=description["successful_episodes"],
        )
        return description

    def _describe_locked(self, record: SessionRecord) -> Dict[str, Any]:
        public_state = "ready" if record.state == "created" else record.state
        return {
            "session_id": record.session_id,
            "run_id": record.run_id,
            "state": public_state,
            "instruction": record.instruction or None,
            "error_code": record.failure_code,
            "benchmark": record.spec.benchmark,
            "task_id": record.spec.task_id,
            "seed": record.spec.seed,
            "episode_length": record.spec.episode_length,
            "observation_level": record.spec.observation_level,
            "bbox_scope": record.spec.bbox_scope,
            "timing": {
                "created_at": utc_iso_from_ns(int(record.created_at_wall * 1e9)),
                "ready_at": (
                    None
                    if record.ready_at_wall is None
                    else utc_iso_from_ns(int(record.ready_at_wall * 1e9))
                ),
                "startup_seconds": record.startup_seconds,
                "completed_at": (
                    None
                    if record.completed_at_wall is None
                    else utc_iso_from_ns(int(record.completed_at_wall * 1e9))
                ),
                "time_to_success_seconds": record.time_to_success_seconds,
            },
        }

    def _remove_record_locked(self, record: SessionRecord) -> None:
        self._sessions.pop(record.session_id, None)
        key = (record.owner, record.idempotency_key)
        current = self._idempotency.get(key)
        if current is not None and current[1] == record.session_id:
            self._idempotency.pop(key, None)

    def _start_record(self, record: SessionRecord, gpu_id: int) -> None:
        child_connection = None
        try:
            with self._lock:
                if record.cancel_requested or self._stopping.is_set():
                    raise _StartCancelled()
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=_episode_process_main,
                args=(child_connection, self.settings, gpu_id, record.spec),
                name=f"libero-session-{record.session_id[-8:]}",
                daemon=True,
            )
            with self._lock:
                record.connection = parent_connection
                record.process = process
                if record.cancel_requested or self._stopping.is_set():
                    raise _StartCancelled()
            try:
                process.start()
            finally:
                child_connection.close()
                child_connection = None
            message = self._wait_message(
                record,
                self.settings.process_start_timeout_seconds,
                WorkerStartTimeout(),
                cancellable=True,
            )
            if message.get("kind") != "ready" or not message.get("ok"):
                raise WorkerStartFailure()
            with self._lock:
                if record.cancel_requested:
                    raise _StartCancelled()
                ready_at = time.monotonic()
                record.ready_at_wall = time.time()
                record.startup_seconds = max(0.0, ready_at - record.created_at)
                if record.run_id is not None:
                    self._evaluation_store.mark_ready(
                        record.session_id,
                        ready_wall_ns=int(record.ready_at_wall * 1e9),
                        startup_ns=int(record.startup_seconds * 1e9),
                    )
                record.state = "created"
                record.instruction = str(message["instruction"])
                record.last_access = ready_at
            self._audit.write(
                "session_created",
                owner=record.owner,
                session_id=record.session_id,
                run_id=record.run_id,
                benchmark=record.spec.benchmark,
                task_id=record.spec.task_id,
                seed=record.spec.seed,
                episode_length=record.spec.episode_length,
                observation_level=record.spec.observation_level,
                bbox_scope=(
                    record.spec.bbox_scope
                    if record.spec.observation_level >= 2
                    else None
                ),
                gpu_id=gpu_id,
                process_id=process.pid,
                created_at=utc_iso_from_ns(int(record.created_at_wall * 1e9)),
                ready_at=utc_iso_from_ns(int(record.ready_at_wall * 1e9)),
                startup_seconds=record.startup_seconds,
            )
        except _StartCancelled:
            self._cleanup_record_resources(record)
            if record.run_id is not None:
                self._evaluation_store.mark_attempt_closed(record.session_id)
            with self._lock:
                record.state = "cancelled"
        except (WorkerStartTimeout, WorkerStartFailure, WorkerDied) as exc:
            confirmed_dead = self._cleanup_record_resources(record)
            if record.run_id is not None:
                self._evaluation_store.mark_attempt_failed(record.session_id, exc.code)
            with self._lock:
                record.state = "failed"
                record.failure_code = exc.code
                record.last_access = time.monotonic()
            self._audit.write(
                "session_start_failed",
                owner=record.owner,
                session_id=record.session_id,
                code=exc.code,
                process_confirmed_dead=confirmed_dead,
            )
        except Exception:
            LOGGER.exception("unexpected dynamic worker start failure")
            confirmed_dead = self._cleanup_record_resources(record)
            if record.run_id is not None:
                try:
                    self._evaluation_store.mark_attempt_failed(
                        record.session_id, WorkerStartFailure.code
                    )
                except Exception:
                    LOGGER.exception("failed to persist attempt startup failure")
            with self._lock:
                record.state = "failed"
                record.failure_code = WorkerStartFailure.code
                record.last_access = time.monotonic()
            self._audit.write(
                "session_start_failed",
                owner=record.owner,
                session_id=record.session_id,
                code=WorkerStartFailure.code,
                process_confirmed_dead=confirmed_dead,
            )
        finally:
            if child_connection is not None:
                try:
                    child_connection.close()
                except Exception:
                    pass
            record.startup_done.set()

    def begin_session(
        self,
        owner: str,
        benchmark: str,
        max_sessions: int,
        *,
        idempotency_key: str,
        request_fingerprint: str,
        run_id: Optional[str] = None,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        episode_length: Optional[int] = None,
        observation_level: int = 1,
        bbox_scope: str = "initial",
    ) -> Dict[str, Any]:
        if benchmark not in BENCHMARK_TASK_COUNTS:
            raise ValueError("unsupported benchmark")
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        if episode_length is None:
            episode_length = self.settings.default_episode_steps

        idempotency_index = (owner, idempotency_key)
        with self._lock:
            if run_id is None:
                try:
                    self._evaluation_store.assert_untracked_session_allowed(owner)
                except StoreError as exc:
                    self._raise_store_error(exc)
                except Exception as exc:
                    LOGGER.exception("failed to check active evaluation Run")
                    raise EvaluationPersistenceFailure() from exc
            existing = self._idempotency.get(idempotency_index)
            if existing is not None:
                existing_fingerprint, existing_session_id = existing
                if existing_fingerprint != request_fingerprint:
                    raise IdempotencyConflict()
                record = self._sessions.get(existing_session_id)
                if record is not None:
                    record.last_access = time.monotonic()
                    return self._describe_locked(record)
                self._idempotency.pop(idempotency_index, None)

            if self._stopping.is_set():
                raise CapacityError()
            owner_count = sum(
                record.owner == owner
                and record.state not in {"failed", "cancelled"}
                for record in self._sessions.values()
            )
            if owner_count >= max_sessions:
                raise QuotaError()
            if task_id is None:
                task_id = secrets.randbelow(BENCHMARK_TASK_COUNTS[benchmark])
            if not 0 <= task_id < BENCHMARK_TASK_COUNTS[benchmark]:
                raise ValueError("task ID is outside the benchmark")
            if seed is None:
                seed = secrets.randbelow(2**32)

            session_id = "ses_" + secrets.token_urlsafe(24)
            slot = self._allocate_slot_locked(session_id)
            spec = SessionSpec(
                benchmark=benchmark,
                task_id=task_id,
                seed=seed,
                episode_length=episode_length,
                observation_level=observation_level,
                bbox_scope=bbox_scope,
            )
            now = time.monotonic()
            record = SessionRecord(
                session_id=session_id,
                owner=owner,
                spec=spec,
                slot_id=slot.slot_id,
                slot_generation=slot.generation,
                state="starting",
                run_id=run_id,
                created_at=now,
                created_at_wall=time.time(),
                last_access=now,
                idempotency_key=idempotency_key,
                request_fingerprint=request_fingerprint,
                phase_successes=[
                    {
                        **phase.public_definition(),
                        "success": False,
                        "first_achieved_step": None,
                    }
                    for phase in phases_for_task(benchmark, task_id)
                ],
            )
            if run_id is not None:
                run_config = {
                    "benchmark": benchmark,
                    "task_id": task_id,
                    "episode_length": episode_length,
                    "observation_level": observation_level,
                    "bbox_scope": bbox_scope,
                }
                try:
                    self._evaluation_store.attach_session(
                        run_id=run_id,
                        owner=owner,
                        session_id=session_id,
                        config=run_config,
                        seed=seed,
                        created_wall_ns=int(record.created_at_wall * 1e9),
                    )
                except StoreError as exc:
                    self._release_slot_locked(record, confirmed_dead=True)
                    self._raise_store_error(exc)
                except Exception as exc:
                    self._release_slot_locked(record, confirmed_dead=True)
                    LOGGER.exception("failed to persist session attachment")
                    raise EvaluationPersistenceFailure() from exc
            self._sessions[session_id] = record
            self._idempotency[idempotency_index] = (
                request_fingerprint,
                session_id,
            )
            startup_thread = threading.Thread(
                target=self._start_record,
                args=(record, slot.gpu_id),
                name=f"libero-start-{session_id[-8:]}",
                daemon=True,
            )
            record.startup_thread = startup_thread
            try:
                startup_thread.start()
            except Exception:
                self._remove_record_locked(record)
                self._release_slot_locked(record, confirmed_dead=True)
                if run_id is not None:
                    self._evaluation_store.mark_attempt_failed(
                        session_id, WorkerStartFailure.code
                    )
                raise WorkerStartFailure()
            return self._describe_locked(record)

    @staticmethod
    def _raise_start_failure(code: Optional[str]) -> None:
        mapping = {
            WorkerStartTimeout.code: WorkerStartTimeout,
            WorkerDied.code: WorkerDied,
            WorkerStartFailure.code: WorkerStartFailure,
        }
        raise mapping.get(code, WorkerStartFailure)()

    def create_session(
        self,
        owner: str,
        benchmark: str,
        max_sessions: int,
        *,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        episode_length: Optional[int] = None,
        observation_level: int = 1,
        bbox_scope: str = "initial",
    ) -> Dict[str, Any]:
        """Synchronous compatibility wrapper used only by the legacy v1 API."""
        idempotency_key = "legacy_" + secrets.token_urlsafe(24)
        created = self.begin_session(
            owner,
            benchmark,
            max_sessions,
            idempotency_key=idempotency_key,
            request_fingerprint=idempotency_key,
            task_id=task_id,
            seed=seed,
            episode_length=episode_length,
            observation_level=observation_level,
            bbox_scope=bbox_scope,
        )
        record = self._record_for_owner(created["session_id"], owner)
        wait_seconds = self.settings.process_start_timeout_seconds + 5
        if not record.startup_done.wait(wait_seconds):
            record.cancel_requested = True
            self._stop_process(record.process)
            raise WorkerStartTimeout()
        with self._lock:
            description = self._describe_locked(record)
            if record.state != "created":
                failure_code = record.failure_code
                self._remove_record_locked(record)
                self._raise_start_failure(failure_code)
            return description

    def _record_for_owner(self, session_id: str, owner: str) -> SessionRecord:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.owner != owner:
                raise NotFoundError()
            return record

    def session_status(self, session_id: str, owner: str) -> Dict[str, Any]:
        record = self._record_for_owner(session_id, owner)
        with self._lock:
            record.last_access = time.monotonic()
            return self._describe_locked(record)

    def _fail_record(self, record: SessionRecord, code: str) -> None:
        confirmed_dead = self._cleanup_record_resources(record)
        if record.run_id is not None:
            try:
                self._evaluation_store.mark_attempt_failed(record.session_id, code)
            except Exception:
                LOGGER.exception("failed to persist attempt failure")
        with self._lock:
            record.state = "failed"
            record.failure_code = code
            record.last_access = time.monotonic()
        self._audit.write(
            "session_failed",
            owner=record.owner,
            session_id=record.session_id,
            code=code,
            process_confirmed_dead=confirmed_dead,
        )

    def _call(self, record: SessionRecord, operation: str, **payload: Any) -> Dict[str, Any]:
        request_id = secrets.token_urlsafe(18)
        if record.connection is None or record.process is None:
            raise WorkerDied()
        try:
            record.connection.send(
                {"request_id": request_id, "operation": operation, **payload}
            )
        except (EOFError, OSError, BrokenPipeError):
            raise WorkerDied()
        message = self._wait_message(
            record, self.settings.request_timeout_seconds, WorkerTimeout()
        )
        if message.get("request_id") != request_id:
            raise WorkerFailure()
        if not message.get("ok"):
            if message.get("error_code") == "INVALID_SESSION_STATE":
                raise ConflictError()
            raise WorkerFailure()
        result = message["result"]
        phase_successes = message.get("phase_successes")
        if isinstance(phase_successes, list):
            with self._lock:
                record.phase_successes = copy.deepcopy(phase_successes)
        if message.get("success_monotonic_ns") is not None:
            result["_gateway_success_monotonic_ns"] = int(
                message["success_monotonic_ns"]
            )
        return result

    @staticmethod
    def _task_progress_locked(record: SessionRecord) -> Dict[str, Any]:
        phases = copy.deepcopy(record.phase_successes)
        completed_phases = sum(phase.get("success") is True for phase in phases)
        total_phases = len(phases)
        return {
            "benchmark": record.spec.benchmark,
            "task_id": record.spec.task_id,
            "completed_phases": completed_phases,
            "total_phases": total_phases,
            "completion_ratio": (
                None if total_phases == 0 else completed_phases / total_phases
            ),
            "phases": phases,
        }

    def task_progress(self, session_id: str, owner: str) -> Dict[str, Any]:
        """Return server-side progress for trusted harness finalization."""

        record = self._record_for_owner(session_id, owner)
        with self._lock:
            return self._task_progress_locked(record)

    def _record_completion_locked(
        self,
        record: SessionRecord,
        *,
        success: bool,
        completion_monotonic_ns: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if record.completion_recorded:
            return None

        parent_monotonic_ns = time.monotonic_ns()
        event_monotonic_ns = (
            parent_monotonic_ns
            if completion_monotonic_ns is None
            else min(completion_monotonic_ns, parent_monotonic_ns)
        )
        completed_wall_ns = time.time_ns() - max(
            0, parent_monotonic_ns - event_monotonic_ns
        )
        session_elapsed_ns = max(
            0, event_monotonic_ns - int(record.created_at * 1_000_000_000)
        )

        run_description: Optional[Dict[str, Any]] = None
        if record.run_id is not None:
            run_started_ns = self._run_started_monotonic_ns.get(record.run_id)
            if run_started_ns is None:
                raise EvaluationPersistenceFailure()
            try:
                run_description = self._evaluation_store.complete_attempt(
                    session_id=record.session_id,
                    success=success,
                    steps=record.steps,
                    completed_wall_ns=completed_wall_ns,
                    session_elapsed_ns=session_elapsed_ns,
                    run_elapsed_ns=max(0, event_monotonic_ns - run_started_ns),
                )
            except Exception as exc:
                LOGGER.exception("failed to persist episode completion")
                raise EvaluationPersistenceFailure() from exc

        record.completion_recorded = True
        record.completed_at_wall = completed_wall_ns / 1_000_000_000
        if success:
            record.time_to_success_seconds = session_elapsed_ns / 1_000_000_000

        event: Dict[str, Any] = {
            "owner": record.owner,
            "session_id": record.session_id,
            "run_id": record.run_id,
            "success": success,
            "steps": record.steps,
            "completed_at": utc_iso_from_ns(completed_wall_ns),
            "session_time_to_success_seconds": record.time_to_success_seconds,
            "first_success": False,
            "task_progress": self._task_progress_locked(record),
        }
        if run_description is not None:
            event["first_success"] = bool(
                success
                and run_description["first_success_session_id"]
                == record.session_id
            )
            event["run_time_to_first_success_seconds"] = run_description[
                "time_to_first_success_seconds"
            ]
        return event

    def _write_completion_audit(self, event: Dict[str, Any]) -> None:
        first_success = bool(event.get("first_success", False))
        episode_event = dict(event)
        episode_event.pop("first_success", None)
        self._audit.write("episode_completed", **episode_event)
        if first_success:
            self._audit.write(
                "evaluation_run_first_success",
                owner=event["owner"],
                run_id=event["run_id"],
                session_id=event["session_id"],
                first_success_at=event["completed_at"],
                time_to_first_success_seconds=event.get(
                    "run_time_to_first_success_seconds"
                ),
            )

    def _operate(
        self,
        session_id: str,
        owner: str,
        operation: str,
        *,
        idempotency_key: Optional[str] = None,
        expected_step_index: Optional[int] = None,
        **payload: Any,
    ) -> Dict[str, Any]:
        record = self._record_for_owner(session_id, owner)
        if not record.operation_lock.acquire(blocking=False):
            raise SessionBusy()
        completion_event: Optional[Dict[str, Any]] = None
        try:
            with self._lock:
                if self._sessions.get(session_id) is not record:
                    raise NotFoundError()
                if operation == "reset" and idempotency_key is not None:
                    if record.reset_idempotency_key == idempotency_key:
                        if record.reset_response is None:
                            raise SessionBusy()
                        record.last_access = time.monotonic()
                        return copy.deepcopy(record.reset_response)
                if operation == "step" and idempotency_key is not None:
                    request_fingerprint = hashlib.sha256(
                        json.dumps(
                            payload.get("action"),
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    if record.step_idempotency_key == idempotency_key:
                        if record.step_request_fingerprint != request_fingerprint:
                            raise IdempotencyConflict()
                        if record.step_response is None:
                            raise SessionBusy()
                        record.last_access = time.monotonic()
                        return copy.deepcopy(record.step_response)
                else:
                    request_fingerprint = None
                if record.state == "starting":
                    raise SessionNotReady()
                if record.state in {"closing", "cancelled", "failed"}:
                    raise ConflictError()
                if (
                    operation == "step"
                    and expected_step_index is not None
                    and expected_step_index != record.steps
                ):
                    raise StepIndexMismatch()
                record.last_access = time.monotonic()
                should_mark_running = (
                    operation == "reset"
                    and record.run_id is not None
                    and record.state == "created"
                )
                if (
                    operation == "reset"
                    and record.completion_recorded
                ):
                    raise ConflictError()
            if should_mark_running:
                try:
                    self._evaluation_store.mark_running(record.session_id)
                except Exception as exc:
                    LOGGER.exception("failed to persist attempt running state")
                    raise EvaluationPersistenceFailure() from exc
            try:
                result = self._call(record, operation, **payload)
            except (WorkerTimeout, WorkerDied, WorkerFailure) as exc:
                self._fail_record(record, exc.code)
                raise
            success_monotonic_ns = result.pop(
                "_gateway_success_monotonic_ns", None
            )
            with self._lock:
                record.last_access = time.monotonic()
                if operation == "reset":
                    record.episode_index += 1
                    record.steps = 0
                    result["episode_index"] = record.episode_index
                    record.reset_idempotency_key = idempotency_key
                    record.reset_response = copy.deepcopy(result)
                    record.step_idempotency_key = None
                    record.step_request_fingerprint = None
                    record.step_response = None
                    record.state = (
                        "completed"
                        if result.get("terminated") or result.get("truncated")
                        else "running"
                    )
                    if record.state == "completed":
                        completion_event = self._record_completion_locked(
                            record,
                            success=bool(result.get("terminated")),
                            completion_monotonic_ns=success_monotonic_ns,
                        )
                elif operation == "step":
                    record.steps += 1
                    record.step_idempotency_key = idempotency_key
                    record.step_request_fingerprint = request_fingerprint
                    record.step_response = copy.deepcopy(result)
                    if result.get("terminated") or result.get("truncated"):
                        record.state = "completed"
                        completion_event = self._record_completion_locked(
                            record,
                            success=bool(result.get("terminated")),
                            completion_monotonic_ns=success_monotonic_ns,
                        )
                elif operation == "result":
                    record.steps = int(result.get("steps", record.steps))
                    result_state = result.get("status")
                    if result_state in {"created", "running", "completed"}:
                        record.state = result_state
                    if result_state == "completed":
                        completion_event = self._record_completion_locked(
                            record, success=bool(result.get("success"))
                        )
            if completion_event is not None:
                self._write_completion_audit(completion_event)
            return result
        finally:
            record.operation_lock.release()

    def reset(
        self,
        session_id: str,
        owner: str,
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self._operate(
            session_id,
            owner,
            "reset",
            idempotency_key=idempotency_key,
        )
        self._audit.write(
            "session_reset",
            owner=owner,
            session_id=session_id,
            episode_index=result["episode_index"],
        )
        return result

    def step(
        self,
        session_id: str,
        owner: str,
        action: List[float],
        *,
        idempotency_key: Optional[str] = None,
        expected_step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._operate(
            session_id,
            owner,
            "step",
            idempotency_key=idempotency_key,
            expected_step_index=expected_step_index,
            action=action,
        )

    def result(self, session_id: str, owner: str) -> Dict[str, Any]:
        record = self._record_for_owner(session_id, owner)
        with self._lock:
            if record.state == "starting":
                raise SessionNotReady()
            if record.state == "failed":
                record.last_access = time.monotonic()
                return {"status": "failed", "success": None, "steps": record.steps}
        return self._operate(session_id, owner, "result")

    def _close_ready_record(self, record: SessionRecord, *, reason: str) -> Dict[str, Any]:
        with self._lock:
            if self._sessions.get(record.session_id) is not record:
                raise NotFoundError()
            record.state = "closing"
        graceful = False
        if (
            record.process is not None
            and record.process.is_alive()
            and record.connection is not None
        ):
            try:
                self._call(record, "close")
                record.process.join(timeout=3)
                graceful = not record.process.is_alive()
            except Exception:
                pass
        confirmed_dead = self._cleanup_record_resources(record)
        if record.run_id is not None:
            try:
                self._evaluation_store.mark_attempt_closed(record.session_id)
            except Exception as exc:
                with self._lock:
                    record.state = "failed"
                    record.failure_code = EvaluationPersistenceFailure.code
                LOGGER.exception("failed to persist attempt close")
                raise EvaluationPersistenceFailure() from exc
        with self._lock:
            self._remove_record_locked(record)
        self._audit.write(
            "session_closed",
            owner=record.owner,
            session_id=record.session_id,
            reason=reason,
            graceful=graceful,
            process_confirmed_dead=confirmed_dead,
            task_progress=self._task_progress_locked(record),
        )
        return {"status": "closed"}

    def close_session(self, session_id: str, owner: str) -> Dict[str, Any]:
        record = self._record_for_owner(session_id, owner)
        with self._lock:
            starting = record.state == "starting"
            if starting:
                record.cancel_requested = True
        if starting:
            process = record.process
            if process is not None and process.is_alive():
                try:
                    process.terminate()
                except Exception:
                    pass
            if not record.startup_done.wait(6):
                raise SessionBusy()
            if record.run_id is not None:
                try:
                    self._evaluation_store.mark_attempt_closed(record.session_id)
                except Exception as exc:
                    LOGGER.exception("failed to persist cancelled attempt close")
                    raise EvaluationPersistenceFailure() from exc
            with self._lock:
                if self._sessions.get(session_id) is record:
                    self._remove_record_locked(record)
            self._audit.write(
                "session_closed",
                owner=owner,
                session_id=session_id,
                reason="cancel_start",
                graceful=False,
                process_confirmed_dead=(
                    record.process is None or not record.process.is_alive()
                ),
                task_progress=self._task_progress_locked(record),
            )
            return {"status": "closed"}

        if not record.operation_lock.acquire(blocking=False):
            raise SessionBusy()
        try:
            return self._close_ready_record(record, reason="client")
        finally:
            record.operation_lock.release()

    def _maintenance_loop(self) -> None:
        while not self._stopping.wait(5.0):
            now = time.monotonic()
            with self._lock:
                records = list(self._sessions.values())
            for record in records:
                if record.state == "starting":
                    continue
                if not record.operation_lock.acquire(blocking=False):
                    continue
                try:
                    with self._lock:
                        if self._sessions.get(record.session_id) is not record:
                            continue
                        expired = (
                            now - record.last_access
                            > self.settings.session_idle_seconds
                        )
                        died = (
                            record.state not in {"failed", "closing", "cancelled"}
                            and record.process is not None
                            and not record.process.is_alive()
                        )
                    if expired:
                        self._close_ready_record(record, reason="idle_timeout")
                    elif died:
                        self._fail_record(record, WorkerDied.code)
                except Exception:
                    LOGGER.exception("session maintenance failed")
                finally:
                    record.operation_lock.release()

    def health(self) -> Dict[str, Any]:
        with self._lock:
            quarantined = sum(slot.quarantined for slot in self._slots)
            in_use = sum(
                slot.assigned_session_id is not None and not slot.quarantined
                for slot in self._slots
            )
            free = sum(
                slot.assigned_session_id is None and not slot.quarantined
                for slot in self._slots
            )
            alive = sum(
                record.process is not None and record.process.is_alive()
                for record in self._sessions.values()
            )
            states: Dict[str, int] = {}
            for record in self._sessions.values():
                state = "ready" if record.state == "created" else record.state
                states[state] = states.get(state, 0) + 1
            return {
                "status": "degraded" if quarantined else "ok",
                "worker_mode": "per_session",
                "workers": alive,
                "active_sessions": len(self._sessions),
                "capacity": len(self._slots) - quarantined,
                "slots": {
                    "total": len(self._slots),
                    "in_use": in_use,
                    "free": free,
                    "quarantined": quarantined,
                },
                "sessions": states,
            }

    def shutdown(self) -> None:
        self._stopping.set()
        with self._lock:
            records = list(self._sessions.values())
            for record in records:
                if record.state == "starting":
                    record.cancel_requested = True
        for record in records:
            if (
                record.state != "starting"
                and record.connection is not None
                and record.process is not None
                and record.process.is_alive()
            ):
                try:
                    request_id = secrets.token_urlsafe(18)
                    record.connection.send(
                        {"request_id": request_id, "operation": "close"}
                    )
                except Exception:
                    pass
            elif record.process is not None and record.process.is_alive():
                try:
                    record.process.terminate()
                except Exception:
                    pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(
            record.process is not None and record.process.is_alive()
            for record in records
        ):
            time.sleep(0.05)
        for record in records:
            self._cleanup_record_resources(record)
        for record in records:
            if record.startup_thread is not None:
                record.startup_thread.join(timeout=1)
        with self._lock:
            self._sessions.clear()
            self._idempotency.clear()
        self._maintenance.join(timeout=1)
