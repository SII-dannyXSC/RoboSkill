from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ACTIVE_ATTEMPT_STATES = ("starting", "ready", "running", "completed")


class StoreError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def utc_iso_from_ns(value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return (
        datetime.fromtimestamp(value / 1_000_000_000, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


class EvaluationStore:
    """Durable, transactionally updated evaluation timing records."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=10000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    launcher_owner TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    label TEXT,
                    config_json TEXT NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    measurement_status TEXT NOT NULL,
                    started_wall_ns INTEGER NOT NULL,
                    finished_wall_ns INTEGER,
                    elapsed_ns INTEGER,
                    first_success_wall_ns INTEGER,
                    time_to_first_success_ns INTEGER,
                    first_success_session_id TEXT,
                    sessions_created INTEGER NOT NULL DEFAULT 0,
                    episodes_completed INTEGER NOT NULL DEFAULT 0,
                    successful_episodes INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(owner, idempotency_key)
                );

                CREATE TABLE IF NOT EXISTS evaluation_attempts (
                    session_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id),
                    owner TEXT NOT NULL,
                    state TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    created_wall_ns INTEGER NOT NULL,
                    ready_wall_ns INTEGER,
                    startup_ns INTEGER,
                    completed_wall_ns INTEGER,
                    session_elapsed_ns INTEGER,
                    success INTEGER,
                    steps INTEGER NOT NULL DEFAULT 0,
                    failure_code TEXT
                );

                CREATE INDEX IF NOT EXISTS attempts_run_idx
                    ON evaluation_attempts(run_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(evaluation_runs)")
            }
            if "launcher_owner" not in columns:
                connection.execute(
                    "ALTER TABLE evaluation_runs ADD COLUMN launcher_owner TEXT"
                )
                connection.execute(
                    "UPDATE evaluation_runs SET launcher_owner=owner "
                    "WHERE launcher_owner IS NULL"
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS runs_launcher_idempotency_idx "
                "ON evaluation_runs(launcher_owner, idempotency_key)"
            )
            now_ns = time.time_ns()
            connection.execute(
                """
                UPDATE evaluation_runs
                SET state='interrupted', measurement_status='interrupted',
                    finished_wall_ns=?,
                    elapsed_ns=MAX(0, ? - started_wall_ns)
                WHERE state='running'
                """,
                (now_ns, now_ns),
            )
            connection.execute(
                """
                UPDATE evaluation_attempts
                SET state='interrupted', failure_code='GATEWAY_RESTARTED'
                WHERE state IN ('starting','ready','running','completed')
                """
            )

    @staticmethod
    def canonical_config(config: Dict[str, Any]) -> str:
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def create_run(
        self,
        *,
        run_id: str,
        owner: str,
        idempotency_key: str,
        request_fingerprint: str,
        label: Optional[str],
        config: Dict[str, Any],
        max_attempts: int,
        started_wall_ns: int,
        launcher_owner: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        config_json = self.canonical_config(config)
        launcher_owner = owner if launcher_owner is None else launcher_owner
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM evaluation_runs
                WHERE launcher_owner=? AND idempotency_key=?
                """,
                (launcher_owner, idempotency_key),
            ).fetchone()
            if existing is not None:
                if existing["request_fingerprint"] != request_fingerprint:
                    raise StoreError("idempotency_conflict")
                return self._describe(connection, existing, None), False
            active = connection.execute(
                "SELECT 1 FROM evaluation_runs WHERE owner=? AND state='running'",
                (owner,),
            ).fetchone()
            if active is not None:
                raise StoreError("active_run_exists")
            connection.execute(
                """
                INSERT INTO evaluation_runs(
                    run_id, owner, launcher_owner, idempotency_key, request_fingerprint,
                    label, config_json, max_attempts, state,
                    measurement_status, started_wall_ns
                ) VALUES(?,?,?,?,?,?,?,?,'running','measuring',?)
                """,
                (
                    run_id,
                    owner,
                    launcher_owner,
                    idempotency_key,
                    request_fingerprint,
                    label,
                    config_json,
                    max_attempts,
                    started_wall_ns,
                ),
            )
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return self._describe(connection, row, 0), True

    def get_run(
        self, run_id: str, owner: str, *, elapsed_ns: Optional[int] = None
    ) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=? AND owner=?",
                (run_id, owner),
            ).fetchone()
            if row is None:
                raise StoreError("run_not_found")
            return self._describe(connection, row, elapsed_ns)

    def get_run_for_launcher(
        self, run_id: str, launcher_owner: str, *, elapsed_ns: Optional[int] = None
    ) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_runs "
                "WHERE run_id=? AND launcher_owner=?",
                (run_id, launcher_owner),
            ).fetchone()
            if row is None:
                raise StoreError("run_not_found")
            return self._describe(connection, row, elapsed_ns)

    def assert_untracked_session_allowed(self, owner: str) -> None:
        with self._lock, self._connect() as connection:
            active = connection.execute(
                "SELECT 1 FROM evaluation_runs WHERE owner=? AND state='running'",
                (owner,),
            ).fetchone()
            if active is not None:
                raise StoreError("run_id_required")

    def attach_session(
        self,
        *,
        run_id: str,
        owner: str,
        session_id: str,
        config: Dict[str, Any],
        seed: int,
        created_wall_ns: int,
    ) -> None:
        config_json = self.canonical_config(config)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=? AND owner=?",
                (run_id, owner),
            ).fetchone()
            if run is None:
                raise StoreError("run_not_found")
            if run["state"] != "running":
                raise StoreError("run_not_active")
            if run["first_success_wall_ns"] is not None:
                raise StoreError("run_already_succeeded")
            if run["config_json"] != config_json:
                raise StoreError("run_config_mismatch")
            if run["sessions_created"] >= run["max_attempts"]:
                raise StoreError("run_attempt_limit")
            connection.execute(
                """
                INSERT INTO evaluation_attempts(
                    session_id, run_id, owner, state, seed, created_wall_ns
                ) VALUES(?,?,?,'starting',?,?)
                """,
                (session_id, run_id, owner, seed, created_wall_ns),
            )
            connection.execute(
                """
                UPDATE evaluation_runs
                SET sessions_created=sessions_created+1
                WHERE run_id=?
                """,
                (run_id,),
            )

    def mark_ready(
        self, session_id: str, *, ready_wall_ns: int, startup_ns: int
    ) -> None:
        with self._lock, self._connect() as connection:
            changed = connection.execute(
                """
                UPDATE evaluation_attempts
                SET state='ready', ready_wall_ns=?, startup_ns=?
                WHERE session_id=? AND state='starting'
                """,
                (ready_wall_ns, startup_ns, session_id),
            ).rowcount
            if changed != 1:
                raise StoreError("attempt_state_conflict")

    def mark_running(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE evaluation_attempts SET state='running'
                WHERE session_id=? AND state='ready'
                """,
                (session_id,),
            )

    def complete_attempt(
        self,
        *,
        session_id: str,
        success: bool,
        steps: int,
        completed_wall_ns: int,
        session_elapsed_ns: int,
        run_elapsed_ns: int,
    ) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = connection.execute(
                "SELECT * FROM evaluation_attempts WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if attempt is None:
                raise StoreError("attempt_not_found")
            if attempt["completed_wall_ns"] is not None:
                run = connection.execute(
                    "SELECT * FROM evaluation_runs WHERE run_id=?",
                    (attempt["run_id"],),
                ).fetchone()
                return self._describe(connection, run, None)
            connection.execute(
                """
                UPDATE evaluation_attempts
                SET state='completed', completed_wall_ns=?,
                    session_elapsed_ns=?, success=?, steps=?
                WHERE session_id=?
                """,
                (
                    completed_wall_ns,
                    session_elapsed_ns,
                    int(success),
                    steps,
                    session_id,
                ),
            )
            connection.execute(
                """
                UPDATE evaluation_runs
                SET episodes_completed=episodes_completed+1,
                    successful_episodes=successful_episodes+?
                WHERE run_id=?
                """,
                (int(success), attempt["run_id"]),
            )
            if success:
                connection.execute(
                    """
                    UPDATE evaluation_runs
                    SET first_success_wall_ns=?,
                        time_to_first_success_ns=?,
                        first_success_session_id=?,
                        measurement_status='first_success_recorded'
                    WHERE run_id=? AND first_success_wall_ns IS NULL
                    """,
                    (
                        completed_wall_ns,
                        run_elapsed_ns,
                        session_id,
                        attempt["run_id"],
                    ),
                )
            run = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=?",
                (attempt["run_id"],),
            ).fetchone()
            return self._describe(connection, run, None)

    def mark_attempt_failed(self, session_id: str, code: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE evaluation_attempts
                SET state='failed', failure_code=?
                WHERE session_id=? AND completed_wall_ns IS NULL
                """,
                (code, session_id),
            )

    def mark_attempt_closed(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE evaluation_attempts
                SET state='closed'
                WHERE session_id=? AND state != 'closed'
                """,
                (session_id,),
            )

    def finish_run(
        self, run_id: str, launcher_owner: str, *, elapsed_ns: int, finished_wall_ns: int
    ) -> Dict[str, Any]:
        placeholders = ",".join("?" for _ in ACTIVE_ATTEMPT_STATES)
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=? AND launcher_owner=?",
                (run_id, launcher_owner),
            ).fetchone()
            if run is None:
                raise StoreError("run_not_found")
            if run["state"] == "finished":
                return self._describe(connection, run, None)
            if run["state"] != "running":
                raise StoreError("run_not_active")
            active = connection.execute(
                f"""
                SELECT 1 FROM evaluation_attempts
                WHERE run_id=? AND state IN ({placeholders}) LIMIT 1
                """,
                (run_id, *ACTIVE_ATTEMPT_STATES),
            ).fetchone()
            if active is not None:
                raise StoreError("run_has_active_sessions")
            measurement_status = (
                "first_success_recorded"
                if run["first_success_wall_ns"] is not None
                else "finished_without_success"
            )
            connection.execute(
                """
                UPDATE evaluation_runs
                SET state='finished', measurement_status=?,
                    finished_wall_ns=?, elapsed_ns=?
                WHERE run_id=?
                """,
                (measurement_status, finished_wall_ns, elapsed_ns, run_id),
            )
            row = connection.execute(
                "SELECT * FROM evaluation_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            return self._describe(connection, row, None)

    def _describe(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        live_elapsed_ns: Optional[int],
    ) -> Dict[str, Any]:
        active = connection.execute(
            """
            SELECT COUNT(*) FROM evaluation_attempts
            WHERE run_id=? AND state IN ('starting','ready','running','completed')
            """,
            (row["run_id"],),
        ).fetchone()[0]
        if row["elapsed_ns"] is not None:
            elapsed_ns = row["elapsed_ns"]
        elif live_elapsed_ns is not None:
            elapsed_ns = live_elapsed_ns
        else:
            elapsed_ns = max(0, time.time_ns() - row["started_wall_ns"])
        return {
            "run_id": row["run_id"],
            "agent_identity": row["owner"],
            "state": row["state"],
            "measurement_status": row["measurement_status"],
            "label": row["label"],
            "config": json.loads(row["config_json"]),
            "max_attempts": row["max_attempts"],
            "started_at": utc_iso_from_ns(row["started_wall_ns"]),
            "finished_at": utc_iso_from_ns(row["finished_wall_ns"]),
            "elapsed_seconds": elapsed_ns / 1_000_000_000,
            "first_success_at": utc_iso_from_ns(row["first_success_wall_ns"]),
            "time_to_first_success_seconds": (
                None
                if row["time_to_first_success_ns"] is None
                else row["time_to_first_success_ns"] / 1_000_000_000
            ),
            "first_success_session_id": row["first_success_session_id"],
            "sessions_created": row["sessions_created"],
            "episodes_completed": row["episodes_completed"],
            "successful_episodes": row["successful_episodes"],
            "active_sessions": active,
            "clock_source": "server_monotonic",
        }
