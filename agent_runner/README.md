# Agent Runner

This layer schedules paper cells. It does not implement robot control and it does not
replace the native CLI Harness.

## Entry points

```bash
# No saved experience
python agent_runner/run.py baseline codex  agent_runner/configs/baseline-codex.example.json
python agent_runner/run.py baseline claude agent_runner/configs/baseline-claude.example.json

# One validated same-task Program Prior
python agent_runner/run.py program-prior codex  agent_runner/configs/program-prior-codex.example.json
python agent_runner/run.py program-prior claude agent_runner/configs/program-prior-claude.example.json

# Text-only/code ablation
python agent_runner/run.py text-only codex  agent_runner/configs/text-only-codex.example.json
python agent_runner/run.py text-only claude agent_runner/configs/text-only-claude.example.json

# Agent-selected Top-K
python agent_runner/run.py topk agent_runner/configs/topk.example.json gpt-5.6-sol
python agent_runner/run.py topk agent_runner/configs/topk.example.json gpt-6-astra
python agent_runner/run.py topk agent_runner/configs/topk.example.json opus
python agent_runner/run.py topk agent_runner/configs/topk.example.json claude-fable-5-1

# E0 -> E1 -> E2 and held-out seeds 100-104
python agent_runner/run.py evolution agent_runner/configs/evolution.example.json codex
python agent_runner/run.py evolution agent_runner/configs/evolution.example.json opus

# Full versus strict tactile/gripper-state ablation
python agent_runner/run.py tactile
```

All examples write only below `.runtime/`. Copy and edit a config before a real
run. Codex supports its normal CLI auth file or a custom provider object; Claude
uses the authentication already configured for the Claude CLI process.

## Generated experience

Run an acquisition config first. Codex exports one selected reviewer under
`experience-reviews/<benchmark>/task-XX/seed-Y/`; Claude exports one under
`experience-reviews/<benchmark>/task-XX/`. The Program Prior schedulers validate
the package, copy it into an immutable batch snapshot, and make the Agent-visible
copy read-only. The checked-in Program Prior examples already point at the output
locations of `acquire-codex.example.json` and `acquire-claude.example.json`; keep the
model used for acquisition and reuse identical.

```bash
cp agent_runner/configs/acquire-codex.example.json .runtime/config/acquire-codex.json
python agent_runner/run.py baseline codex .runtime/config/acquire-codex.json

cp agent_runner/configs/acquire-claude.example.json .runtime/config/acquire-claude.json
python agent_runner/run.py baseline claude .runtime/config/acquire-claude.json
```

For Top-K, acquire seed-0 experience separately for all four paper models, then
freeze those outputs into the shared library layout:

```bash
python agent_runner/experiments/topk/prepare_libraries.py \
  --output-root .runtime/skill-libraries \
  --gpt-5-6-sol-root .runtime/paper/acquire-sol/batches/acquire-sol/experience-reviews/libero_10 \
  --gpt-6-astra-root .runtime/paper/acquire-astra/batches/acquire-astra/experience-reviews/libero_10 \
  --opus-root .runtime/paper/acquire-opus/batches/acquire-opus/experience-reviews/libero_10 \
  --fable-root .runtime/paper/acquire-fable/batches/acquire-fable/experience-reviews/libero_10
```

The two Codex roots contain one `seed-*` directory per task; the Claude roots contain
the package directly below each task. The preparer validates every package and
refuses ambiguous or existing destinations.

The text-only source builder accepts explicit acquisition batch roots:

```bash
python agent_runner/experiments/text_only/prepare_text_action_sources.py \
  --codex-batch-root .runtime/path/to/codex/batch \
  --codex-target .runtime/text-action/codex \
  --claude-batch-root .runtime/path/to/claude/batch \
  --claude-target .runtime/text-action/claude
```

For the other-task condition, prepare the fixed bidirectional mapping
`0↔1, 2↔8, 3↔5, 4↔9, 6↔7`, then launch the paired queues. No target-task package is
added in this condition:

```bash
python agent_runner/experiments/cross_task/prepare_mapped_sources.py \
  --output-root .runtime/mapped-sources \
  --claude-program-root .runtime/paper/acquire-claude/batches/acquire-claude/experience-reviews \
  --codex-program-root .runtime/paper/acquire-codex/batches/acquire-codex/experience-reviews \
  --claude-text-action-root .runtime/text-action/claude \
  --codex-text-action-root .runtime/text-action/codex
python agent_runner/experiments/cross_task/interleaved_queue.py codex
python agent_runner/experiments/cross_task/interleaved_queue.py claude
```

`catalog.json` is the machine-readable protocol for the settings published here. It
records the current paper's seed sets and selections. The historical cross-agent Kimi
setting and its DeepSeek-Harness executor are deliberately not included.

## Stop and resume

Schedulers handle SIGINT/SIGTERM, signal all running cell process groups, wait for
cleanup, and persist terminal state. Use `--resume-existing` only with the same
config and an existing state file. A native CLI exit within a cell is handled by the
Harness and does not by itself terminate the cell.
