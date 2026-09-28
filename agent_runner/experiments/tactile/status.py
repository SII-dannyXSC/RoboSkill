#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
STATE_ROOT = Path(
    os.environ.get("TACTILE_STATE_DIR", REPO_ROOT / ".runtime/paper/tactile")
).expanduser().resolve()


def read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


suite = read(STATE_ROOT / "suite-state.json")
print(
    f"suite={suite.get('status', 'not-started')} "
    f"phase={suite.get('phase', '-')} network_limit={suite.get('network_ratio_limit', 0.05):.0%}"
)
for condition in ("full", "strict"):
    spec = read(HERE / f"{condition}.json")
    root = Path(spec.get("experiment_root", ".")).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    state_path = root.resolve() / "batches" / str(spec.get("batch_id", "-")) / "batch-state.json"
    state = read(state_path)
    counts = Counter(job.get("state", "unknown") for job in state.get("jobs", []))
    print(
        f"{condition}: status={state.get('status', 'not-started')} "
        f"cells={len(state.get('jobs', []))}/50 states={dict(counts)}"
    )
network = read(STATE_ROOT / "network-state.json")
if network:
    print(
        f"network: status={network.get('status')} phase={network.get('phase')} "
        f"flagged={network.get('flagged_cells', 0)} blocked={network.get('blocked_cells', 0)}"
    )
