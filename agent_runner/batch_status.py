#!/usr/bin/env python3
"""Print a scheduler state summary without contacting the Agent or Gateway."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain one JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path, help="batch-state.json or experiment-state.json")
    args = parser.parse_args()
    path = args.state.expanduser().resolve()
    state = load_object(path)
    jobs = state.get("jobs", [])
    counts = Counter(str(job.get("state", "unknown")) for job in jobs)
    print(
        json.dumps(
            {
                "state": str(path),
                "batch_id": state.get("batch_id"),
                "status": state.get("status"),
                "scheduler_pid": state.get("scheduler_pid"),
                "jobs": len(jobs),
                "summary": dict(sorted(counts.items())),
                "updated_at": state.get("updated_at"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
