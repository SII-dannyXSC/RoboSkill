#!/usr/bin/env python3
"""Finish the current server-timed Run and persist its authoritative timing."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from libero_client import LiberoAPIError, LiberoClient


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main() -> int:
    run_id = required("LIBERO_RUN_ID")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise SystemExit("LIBERO_RUN_ID contains unsafe path characters")

    client = LiberoClient(
        required("LIBERO_API_URL"),
        required("LIBERO_API_TOKEN"),
    )
    try:
        result = client.finish_evaluation_run(run_id)
    except LiberoAPIError as exc:
        print(f"Could not finish Evaluation Run: {exc}", file=sys.stderr)
        return 1

    recorded = dict(result)
    recorded["client_finalized_at"] = datetime.now(timezone.utc).isoformat()
    output = Path.cwd() / "artifacts" / run_id / "evaluation_timing.json"
    atomic_json(output, recorded)

    print(f"LIBERO Run finalized: {run_id}")
    print(f"measurement_status={result.get('measurement_status')}")
    print(f"elapsed_seconds={result.get('elapsed_seconds')}")
    print(
        "time_to_first_success_seconds="
        f"{result.get('time_to_first_success_seconds')}"
    )
    print(f"saved={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
