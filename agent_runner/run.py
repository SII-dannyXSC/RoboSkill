#!/usr/bin/env python3
"""Dispatch one published experiment runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCHEDULERS = HERE / "schedulers"
EXPERIMENTS = HERE / "experiments"


def exec_python(script: Path, arguments: list[str], environment: dict[str, str] | None = None) -> None:
    if not script.is_file():
        raise SystemExit(f"runner is missing: {script}")
    env = os.environ.copy()
    if environment:
        env.update(environment)
    os.execve(sys.executable, [sys.executable, str(script), *arguments], env)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for setting in ("baseline", "program-prior", "text-only"):
        sub = subparsers.add_parser(setting)
        sub.add_argument("executor", choices=("codex", "claude"))
        sub.add_argument("config", type=Path)
        sub.add_argument("--resume-existing", action="store_true")

    topk = subparsers.add_parser("topk")
    topk.add_argument("config", type=Path)
    topk.add_argument("model_key")
    topk.add_argument("--resume-existing", action="store_true")

    evolution = subparsers.add_parser("evolution")
    evolution.add_argument("config", type=Path)
    evolution.add_argument("model_key", choices=("codex", "opus"))
    evolution.add_argument("--resume-existing", action="store_true")

    tactile = subparsers.add_parser("tactile")
    tactile.add_argument("--full-config", type=Path, default=EXPERIMENTS / "tactile/full.json")
    tactile.add_argument("--strict-config", type=Path, default=EXPERIMENTS / "tactile/strict.json")

    args = parser.parse_args()
    if args.command == "tactile":
        exec_python(
            EXPERIMENTS / "tactile/orchestrate.py",
            [],
            {
                "TACTILE_FULL_CONFIG": str(args.full_config.expanduser().resolve()),
                "TACTILE_STRICT_CONFIG": str(args.strict_config.expanduser().resolve()),
            },
        )

    if args.command == "topk":
        values = [str(args.config.expanduser().resolve()), args.model_key]
        if args.resume_existing:
            values.append("--resume-existing")
        exec_python(EXPERIMENTS / "topk/orchestrator.py", values)

    if args.command == "evolution":
        values = [str(args.config.expanduser().resolve()), args.model_key]
        if args.resume_existing:
            values.append("--resume-existing")
        exec_python(EXPERIMENTS / "evolution/model_orchestrator.py", values)

    table = {
        ("baseline", "codex"): SCHEDULERS / "codex_batch.py",
        ("baseline", "claude"): SCHEDULERS / "claude_batch.py",
        ("program-prior", "codex"): SCHEDULERS / "codex_program_prior_batch.py",
        ("program-prior", "claude"): SCHEDULERS / "claude_program_prior_batch.py",
        ("text-only", "codex"): SCHEDULERS / "codex_text_action_batch.py",
        ("text-only", "claude"): SCHEDULERS / "claude_text_action_batch.py",
    }
    values = [str(args.config.expanduser().resolve())]
    if args.resume_existing:
        values.append("--resume-existing")
    exec_python(table[(args.command, args.executor)], values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
