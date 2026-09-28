#!/usr/bin/env python3
"""Create a non-interactive LIBERO path config below the ignored runtime root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    benchmark_root = source / "libero/libero"
    required = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "assets": benchmark_root / "assets",
    }
    missing = [str(path) for path in required.values() if not path.is_dir()]
    if missing:
        raise SystemExit(f"LIBERO source is incomplete; missing directories: {missing}")
    config_root = args.config_root.expanduser().resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    values = {
        **{name: str(path) for name, path in required.items()},
        "datasets": str(source / "datasets"),
    }
    path = config_root / "config.yaml"
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"libero_config={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
