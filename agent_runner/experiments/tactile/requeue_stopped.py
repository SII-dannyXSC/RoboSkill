#!/usr/bin/env python3
"""Requeue only cells cancelled by a deliberate scheduler concurrency restart."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SPEC = HERE / "strict.json"


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    root = Path(spec["experiment_root"]).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    state_path = (
        root.resolve()
        / "batches"
        / spec["batch_id"]
        / "batch-state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    changed = 0
    for job in state["jobs"]:
        if job.get("state") != "cancelled" or job.get("outcome_reason") != "stopped":
            continue
        job.setdefault("previous_attempts", []).append(
            {
                "workspace": job.get("workspace"),
                "pid": job.get("pid"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "reason": "concurrency-restart",
            }
        )
        job.update(
            state="queued",
            pid=None,
            workspace=None,
            started_at=None,
            finished_at=None,
            outcome_reason=None,
            reviewer=False,
            review_status="",
        )
        changed += 1
    state.update(
        status="stopped",
        max_concurrency=int(spec["max_concurrency"]),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    temporary = state_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, state_path)
    print(f"requeued={changed} state={state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
