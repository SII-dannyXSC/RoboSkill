#!/usr/bin/env python3
"""Create and optionally launch one portable Harness/LIBERO attempt."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from libero_runtime import atomic_bytes, atomic_json
from prompt_context import DEFAULT_DEADLINE_SECONDS


PROTOTYPE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ASSETS = ("libero_sdk.py",)
AGENT_API_GATE_MARKER = "{{LIBERO_GATE_POLICY}}"
PROMPT_ASSET_BY_WORKFLOW = {
    "plain": "AGENT_PROMPT_PLAIN.md",
    "operation-ledger": "AGENT_PROMPT.md",
    "trusted-step-liveness": "AGENT_PROMPT.md",
    "episode-ledger": "AGENT_PROMPT_LEDGER.md",
    "action-ledger": "AGENT_PROMPT_LEDGER.md",
    "research-explorer": "AGENT_PROMPT_RESEARCH_EXPLORER.md",
    "research-executor": "AGENT_PROMPT_RESEARCH_EXECUTOR.md",
}
ATTEMPT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
DEFAULT_MODEL_OUTAGE_GRACE_SECONDS = 900
DEFAULT_ACTIVE_RESPONSE_TIMEOUT_SECONDS = 180
DEFAULT_MAX_ACTIVE_RESPONSE_TIMEOUTS = 3
DEFAULT_PREPARE_SOFT_LIMIT_SECONDS = 600
DEFAULT_PREPARE_HARD_LIMIT_SECONDS = 1200
NO_TACTILE_PROFILE = "l4-no-tactile-no-gripper-proprio-v1"


def default_attempt_id() -> str:
    """Return a readable attempt id with enough entropy for concurrent launchers."""

    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return f"attempt-{stamp}-{uuid.uuid4().hex[:6]}"


def prepare_attempt(
    *,
    experiment_root: Path,
    benchmark: str,
    task_id: int,
    level: int,
    attempt_id: str,
    episode_length: int,
    bbox_scope: str,
    seed: int | None,
    deadline_seconds: int,
    gateway_url: str,
    harness_url: str,
    provider: str,
    model: str,
    agent_preset: str,
    workflow_variant: str = "operation-ledger",
    continue_on_unsuccessful_turn: bool = True,
    model_outage_grace_seconds: int = DEFAULT_MODEL_OUTAGE_GRACE_SECONDS,
    active_response_timeout_seconds: int | None = DEFAULT_ACTIVE_RESPONSE_TIMEOUT_SECONDS,
    max_active_response_timeouts: int = DEFAULT_MAX_ACTIVE_RESPONSE_TIMEOUTS,
    prepare_soft_limit_seconds: int = DEFAULT_PREPARE_SOFT_LIMIT_SECONDS,
    prepare_hard_limit_seconds: int = DEFAULT_PREPARE_HARD_LIMIT_SECONDS,
    label: str | None = None,
    frozen_ledger_file: Path | None = None,
    experience_ledger_file: Path | None = None,
    program_ledger_file: Path | None = None,
    program_selection: dict[str, Any] | None = None,
    observation_profile: str = NO_TACTILE_PROFILE,
) -> tuple[Path, Path, str]:
    """Create a new attempt Workspace and return workspace/config/session id."""

    if task_id < 0:
        raise ValueError("task_id must be non-negative")
    if level not in {1, 2, 3, 4}:
        raise ValueError("level must be 1, 2, 3, or 4")
    if level != 4 or observation_profile != NO_TACTILE_PROFILE:
        raise ValueError(
            f"this frozen runtime requires level=4 and observation_profile={NO_TACTILE_PROFILE}"
        )
    if episode_length < 1:
        raise ValueError("episode_length must be positive")
    if deadline_seconds < 1:
        raise ValueError("deadline_seconds must be positive")
    if not isinstance(continue_on_unsuccessful_turn, bool):
        raise ValueError("continue_on_unsuccessful_turn must be boolean")
    if (
        not isinstance(model_outage_grace_seconds, int)
        or isinstance(model_outage_grace_seconds, bool)
        or model_outage_grace_seconds < 1
    ):
        raise ValueError("model_outage_grace_seconds must be positive")
    for name, value in {"max_active_response_timeouts": max_active_response_timeouts}.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive")
    if active_response_timeout_seconds is not None and (
        not isinstance(active_response_timeout_seconds, int)
        or isinstance(active_response_timeout_seconds, bool)
        or active_response_timeout_seconds < 1
    ):
        raise ValueError("active_response_timeout_seconds must be positive or None")
    if prepare_soft_limit_seconds < 1 or prepare_hard_limit_seconds <= prepare_soft_limit_seconds:
        raise ValueError("prepare limits require 0 < soft < hard")
    if bbox_scope not in {"initial", "every_frame"}:
        raise ValueError("bbox_scope must be initial or every_frame")
    if workflow_variant not in {
        "plain",
        "operation-ledger",
        "trusted-step-liveness",
        "episode-ledger",
        "action-ledger",
        "research-explorer",
        "research-executor",
    }:
        raise ValueError(
            "workflow_variant must be plain, operation-ledger, trusted-step-liveness, episode-ledger, action-ledger, research-explorer, or research-executor"
        )
    if seed is not None and not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must fit uint32")
    if not ATTEMPT_ID_PATTERN.fullmatch(attempt_id) or attempt_id in {".", ".."}:
        raise ValueError("attempt_id must be one safe path component")
    for name, value in {
        "benchmark": benchmark,
        "gateway_url": gateway_url,
        "harness_url": harness_url,
        "provider": provider,
        "model": model,
        "agent_preset": agent_preset,
    }.items():
        if not value.strip():
            raise ValueError(f"{name} must be non-empty")

    root = experiment_root.expanduser().resolve()
    workspace = root / "runs" / f"task-{task_id:02d}" / f"level-{level}" / attempt_id
    workspace.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name in WORKSPACE_ASSETS:
        source = PROTOTYPE_ROOT / name
        atomic_bytes(workspace / name, source.read_bytes())
    agent_api = (PROTOTYPE_ROOT / "AGENT_API.md").read_text(encoding="utf-8")
    gate_policy = (
        "Controller 在 Agent 开始前执行且仅执行一次初始 Reset，并将该 Observation 持久化为 Episode 1 的首帧。"
        "`POST reset` 和 `POST step` 从任务开始即直接可用。"
        "不要用 Reset 获取初始 Observation；只有确实需要开始新 Episode 时才调用 Reset。"
        if workflow_variant in {"plain", "research-explorer", "research-executor"}
        else
        "Controller 在 Agent 开始前执行且仅执行一次初始 Reset，并将该 Observation 持久化为 Episode 1 的首帧。"
        "何时允许 `POST reset` 或 `POST step` 由本 trial 注入的 Harness workflow 决定；被拒绝时代理返回 HTTP 409 和当前 gate 原因。"
        "不要用 Reset 获取初始 Observation，也不要通过反复调用接口猜测 gate。"
        "账本版在 `episode_ledger_checkpoint(action=\"start_task\", ...)` 被接受后开放首个 Step；"
        "后续 Reset 会暂时关闭 Step，直到逐条完成 episode 经验迁移。"
    )
    if agent_api.count(AGENT_API_GATE_MARKER) != 1:
        raise RuntimeError("AGENT_API.md must contain exactly one gate-policy marker")
    atomic_bytes(
        workspace / "AGENT_API.md",
        agent_api.replace(AGENT_API_GATE_MARKER, gate_policy).encode("utf-8"),
    )
    prompt_source = PROTOTYPE_ROOT / PROMPT_ASSET_BY_WORKFLOW[workflow_variant]
    atomic_bytes(workspace / "AGENT_PROMPT.md", prompt_source.read_bytes())

    session_id = str(uuid.uuid4())
    experiment: dict[str, Any] = {
        "label": label or f"{benchmark}-task{task_id:02d}-level{level}-{attempt_id}",
        "task": {"benchmark": benchmark, "task_id": task_id},
        "episode_length": episode_length,
        "observation": {
            "level": level,
            "bbox_scope": bbox_scope,
            "profile": observation_profile,
        },
    }
    if seed is not None:
        experiment["seed"] = seed
    config = {
        # Absolute paths make the audit snapshot unambiguous. controller.py
        # also accepts a relative workspace resolved against this config file.
        "workspace": str(workspace),
        "gateway_url": gateway_url.rstrip("/"),
        "deadline_seconds": deadline_seconds,
        "experiment": experiment,
        "harness": {
            "url": harness_url.rstrip("/"),
            "session_id": session_id,
            "provider": provider,
            "model": model,
            "title": f"LIBERO task {task_id} level {level} {attempt_id}",
            "agent_preset": agent_preset,
            "workflow_variant": workflow_variant,
            "rpc_timeout_seconds": 10,
            "prompt_file": "AGENT_PROMPT.md",
            "continue_on_unsuccessful_turn": continue_on_unsuccessful_turn,
            "active_response_timeout_seconds": active_response_timeout_seconds,
            "max_active_response_timeouts": max_active_response_timeouts,
            "prepare_soft_limit_seconds": prepare_soft_limit_seconds,
            "prepare_hard_limit_seconds": prepare_hard_limit_seconds,
            "transient_recovery": {
                "initial_backoff_seconds": 5,
                "max_backoff_seconds": 120,
                "max_outage_seconds": model_outage_grace_seconds,
                "healthy_confirmations": 2,
                "confirmation_interval_seconds": 2,
            },
        },
    }
    if workflow_variant == "research-executor":
        if frozen_ledger_file is None:
            raise ValueError("research-executor requires frozen_ledger_file")
        config["harness"]["frozen_ledger_file"] = str(
            frozen_ledger_file.expanduser().resolve()
        )
    elif frozen_ledger_file is not None:
        raise ValueError("frozen_ledger_file is only valid for research-executor")
    for field, ledger_file in {
        "experience_ledger_file": experience_ledger_file,
        "program_ledger_file": program_ledger_file,
    }.items():
        if ledger_file is None:
            continue
        if workflow_variant not in {"episode-ledger", "action-ledger"}:
            raise ValueError(f"{field} requires episode-ledger or action-ledger")
        config["harness"][field] = str(ledger_file.expanduser().resolve())
    if program_selection is not None:
        if program_ledger_file is None:
            raise ValueError("program_selection requires program_ledger_file")
        config["harness"]["program_selection"] = program_selection
    config_path = workspace / "config.json"
    atomic_json(config_path, config)
    return workspace, config_path, session_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a fresh attempt under any experiment root and launch it"
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--episode-length", type=int, default=2000)
    parser.add_argument("--bbox-scope", choices=("initial", "every_frame"), default="initial")
    parser.add_argument(
        "--observation-profile",
        choices=(NO_TACTILE_PROFILE,),
        default=NO_TACTILE_PROFILE,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deadline-seconds", type=int, default=DEFAULT_DEADLINE_SECONDS)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:18080")
    parser.add_argument("--harness-url", default="http://127.0.0.1:8080")
    parser.add_argument("--provider", default="configured-agent")
    parser.add_argument("--model", default="configured-model")
    parser.add_argument("--agent-preset", default="standard")
    parser.add_argument(
        "--workflow-variant",
        choices=(
            "plain", "operation-ledger", "trusted-step-liveness", "episode-ledger", "action-ledger",
            "research-explorer", "research-executor",
        ),
        default="operation-ledger",
    )
    parser.add_argument("--experience-ledger-file", type=Path)
    parser.add_argument("--program-ledger-file", type=Path)
    parser.add_argument("--program-id")
    parser.add_argument("--program-version", type=int)
    parser.add_argument("--program-prior-id")
    parser.add_argument(
        "--disable-active-response-timeout",
        action="store_true",
        help="Disable post-PLAN response timeout while retaining the total task deadline",
    )
    parser.add_argument("--frozen-ledger-file", type=Path, default=None)
    parser.add_argument("--prepare-soft-limit-seconds", type=int, default=DEFAULT_PREPARE_SOFT_LIMIT_SECONDS)
    parser.add_argument("--prepare-hard-limit-seconds", type=int, default=DEFAULT_PREPARE_HARD_LIMIT_SECONDS)
    parser.add_argument(
        "--continue-on-unsuccessful-turn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue the same Harness Session when a Turn ends before success",
    )
    parser.add_argument(
        "--model-outage-grace-seconds",
        type=int,
        default=DEFAULT_MODEL_OUTAGE_GRACE_SECONDS,
        help="Keep the same Harness/LIBERO Session this long during transient model outages",
    )
    parser.add_argument(
        "--active-response-timeout-seconds",
        type=int,
        default=DEFAULT_ACTIVE_RESPONSE_TIMEOUT_SECONDS,
        help="Cancel and continue one post-PLAN model response after this wall time",
    )
    parser.add_argument(
        "--max-active-response-timeouts",
        type=int,
        default=DEFAULT_MAX_ACTIVE_RESPONSE_TIMEOUTS,
        help="Fail an attempt after this many recovered post-PLAN response timeouts",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create the Workspace/config but do not allocate or launch",
    )
    args = parser.parse_args()
    if (args.program_id is None) != (args.program_version is None):
        parser.error("--program-id and --program-version must be provided together")
    if args.program_prior_id is not None and args.program_id is None:
        parser.error("--program-prior-id requires --program-id and --program-version")

    workspace, config_path, session_id = prepare_attempt(
        experiment_root=args.experiment_root,
        benchmark=args.benchmark,
        task_id=args.task_id,
        level=args.level,
        attempt_id=args.attempt_id or default_attempt_id(),
        episode_length=args.episode_length,
        bbox_scope=args.bbox_scope,
        observation_profile=args.observation_profile,
        seed=args.seed,
        deadline_seconds=args.deadline_seconds,
        gateway_url=args.gateway_url,
        harness_url=args.harness_url,
        provider=args.provider,
        model=args.model,
        agent_preset=args.agent_preset,
        workflow_variant=args.workflow_variant,
        continue_on_unsuccessful_turn=args.continue_on_unsuccessful_turn,
        model_outage_grace_seconds=args.model_outage_grace_seconds,
        active_response_timeout_seconds=None if args.disable_active_response_timeout else args.active_response_timeout_seconds,
        max_active_response_timeouts=args.max_active_response_timeouts,
        prepare_soft_limit_seconds=args.prepare_soft_limit_seconds,
        prepare_hard_limit_seconds=args.prepare_hard_limit_seconds,
        label=args.label,
        frozen_ledger_file=args.frozen_ledger_file,
        experience_ledger_file=args.experience_ledger_file,
        program_ledger_file=args.program_ledger_file,
        program_selection=(
            {
                "program_id": args.program_id,
                "version": args.program_version,
                **(
                    {"prior_id": args.program_prior_id}
                    if args.program_prior_id is not None
                    else {}
                ),
            }
            if args.program_id is not None and args.program_version is not None
            else None
        ),
    )
    print(f"workspace={workspace}", flush=True)
    print(f"config={config_path}", flush=True)
    print(f"harness_session_id={session_id}", flush=True)
    if args.prepare_only:
        return 0

    controller = PROTOTYPE_ROOT / "controller.py"
    os.execv(sys.executable, [sys.executable, str(controller), str(config_path)])
    raise AssertionError("os.execv returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
