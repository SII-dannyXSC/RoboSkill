#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXPECTED_MAP = {"0": 1, "1": 0, "2": 8, "8": 2, "3": 5, "5": 3, "4": 9, "9": 4, "6": 7, "7": 6}


def main() -> int:
    total = 0
    for name in ("claude-program", "claude-text-action", "codex-program", "codex-text-action"):
        spec = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
        assert spec["source_task_map"]["libero_10"] == EXPECTED_MAP
        assert spec["task_groups"] == [{"benchmark": "libero_10", "task_ids": list(range(10))}]
        assert spec["seeds"] == list(range(5))
        assert spec["concurrency"]["initial"] == 1
        assert spec["concurrency"]["maximum"] == 1
        cells = len(spec["task_groups"][0]["task_ids"]) * len(spec["seeds"])
        assert cells == 50
        total += cells
    assert total == 200
    print("validated: 4 conditions, 50 cells each, 200 total, concurrency 1/model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
