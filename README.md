# Explore, Execute, Evolve — Reproduction Code

This repository is the source-only release of the native Agent CLI execution path used
for the paper. It has three top-level components:

```text
libero/         LIBERO Gateway, trusted SDK/proxy, and observation profiles
harness/        One-Agent lifecycle for native Codex CLI and Claude Code
agent_runner/   Batch scheduling and paper experiment settings
```

The release does not contain run workspaces, logs, sessions, videos, credentials,
historical results, generated experience packages, or generated skills. It also does
not ship the older DeepSeek Harness, Kimi provider integration, or the standalone
portable ledger/MCP implementation.

## Execution boundary

`libero/` owns simulator information and exposes only the Agent-facing SDK.
`harness/` owns one Agent process and one LIBERO lease. `agent_runner/` expands
task/seed matrices, limits concurrency, retries structural startup failures, and
starts one Harness controller per cell.

A native CLI process exiting is not normally the end of a cell. If the Gateway has
not confirmed success and the four-hour deadline has not expired, the Harness resumes
the same Codex thread or Claude session. A cell ends on server-confirmed success,
deadline expiry, an explicit stop signal, or an unrecoverable controller error.
Process groups are shut down with SIGINT, then SIGTERM, then SIGKILL.

## Setup

The Gateway and simulator use the pinned LIBERO environment:

```bash
make bootstrap
conda activate agent-for-robot-libero
```

Install and authenticate the Agent CLI you intend to use separately. No API key is
stored in this repository.

Start the real simulator Gateway in the activated environment. Set the GPU list and
per-GPU capacity for the machine before launching it:

```bash
export LIBERO_GATEWAY_GPU_IDS=0,1,2,3
export LIBERO_GATEWAY_MAX_SESSIONS_PER_GPU=1
./scripts/run_libero_gateway.sh
```

The Gateway binds to loopback by default. Runtime credentials, audit records, and its
SQLite state are generated only below ignored `.runtime/`.

Copy one configuration without modifying the checked-in example:

```bash
mkdir -p .runtime/config
cp agent_runner/configs/baseline-codex.example.json .runtime/config/baseline-codex.json
# Edit executable, model, provider/auth, concurrency, and output root.
```

Start a batch after the Gateway is healthy at the configured URL:

```bash
python agent_runner/run.py baseline codex .runtime/config/baseline-codex.json
python agent_runner/batch_status.py \
  .runtime/paper/baseline-codex/batches/baseline-codex/batch-state.json
```

Claude uses the parallel entry:

```bash
cp agent_runner/configs/baseline-claude.example.json .runtime/config/baseline-claude.json
python agent_runner/run.py baseline claude .runtime/config/baseline-claude.json
```

## Experience acquisition and reuse

The acquisition configs enable the same-session post-success reviewer. The reviewer
can only read the completed run and must produce a validated action ledger, keyframes,
scene-local binding, summary, and byte-identical copies of code that actually ran.

```bash
cp agent_runner/configs/acquire-codex.example.json .runtime/config/acquire-codex.json
# Set the exact paper model in the copied config.
python agent_runner/run.py baseline codex .runtime/config/acquire-codex.json

cp agent_runner/configs/acquire-claude.example.json .runtime/config/acquire-claude.json
# Set the exact paper model in the copied config.
python agent_runner/run.py baseline claude .runtime/config/acquire-claude.json

python agent_runner/run.py program-prior codex \
  agent_runner/configs/program-prior-codex.example.json
```

Generated packages stay below ignored `.runtime/` paths and are never release inputs.
See [agent_runner/README.md](agent_runner/README.md) for same-task, Top-K,
other-task, evolution, text-only, and tactile commands.

## Latest tactile setting

The published Table 10 setting is GPT-6 Astra on all ten LIBERO-10 tasks with seeds
0–4 (50 cells per condition). The strict condition is
`l4-no-tactile-no-gripper-proprio-v1`: force-related fields and gripper state are
removed, while the simulator and seven-dimensional action space are unchanged.

```bash
python agent_runner/run.py tactile \
  --full-config agent_runner/experiments/tactile/full.json \
  --strict-config agent_runner/experiments/tactile/strict.json
```

The older T6/T9 Sol/Opus tactile rerun is not the paper setting and is not the
published path.

## Verification

```bash
make smoke
make verify-publication
```

Source lineage and pre-relocation hashes are recorded in
[PROVENANCE.md](PROVENANCE.md). Exact numerical reproduction additionally requires
access to the named model versions and equivalent GPU resources.
