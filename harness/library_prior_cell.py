#!/usr/bin/env python3
"""Run a native CLI cell with a retrieved catalog from a ten-skill library."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


LIBRARY_INSTRUCTION = """

本 Workspace 的 prior 是从十任务经验库按当前指令检索出的候选目录，不是预先合并好的单一程序。
先读取顶层 program/program-index.json，再分别读取 candidates 中列出的经验索引。候选均为独立、
只读、已验证的成功经验。请根据当前 Reset 的真实观测逐模块选择：可以从不同候选分别采用感知、
抓取、运输、放置、验证或恢复方法，也可以放弃不适用内容。不要盲目复制整包，也不要修改候选。
所有真正使用或适配的代码必须先复制到 agent/，数值参数必须根据当前 RGB、深度、标定、力反馈和
本体状态重新计算。经验只提供方法，不提供当前场景事实。
"""


def load_module(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("library_prior_base_cell", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import cell: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("executor", choices=("codex", "claude"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    json.loads(config_path.read_text(encoding="utf-8"))
    if args.executor == "codex":
        path = REPO_ROOT / "harness/codex/evolution_cell.py"
    else:
        path = REPO_ROOT / "harness/claude/program_prior_cell.py"
    module = load_module(path)
    module.INITIAL_PROMPT += LIBRARY_INSTRUCTION
    sys.argv = [str(path), str(config_path)]
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
