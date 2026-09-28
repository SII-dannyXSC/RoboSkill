# Native Agent Harness

This directory contains the single-cell controllers copied from the paper experiment
tree. It is not a new portable Harness and it does not start an Agent service.

- `codex/cell.py`: baseline/acquisition controller using `codex exec --json`.
- `codex/program_prior_cell.py`: filesystem Program Prior.
- `codex/text_action_cell.py`: text plus action-trajectory ablation.
- `codex/evolution_cell.py`: reuse followed by same-session evolution review.
- `claude/cell.py`: baseline/acquisition controller with a fixed Claude session id.
- `claude/program_prior_cell.py` and `text_action_cell.py`: matching reuse paths.
- `library_prior_cell.py`: adds the Top-K library instruction, then calls the
  native Codex or Claude prior controller.

Each controller allocates one trusted LIBERO lease, persists the first Reset, exposes
only a localhost lease proxy to the Agent workspace, starts the native CLI, polls the
Gateway's authoritative result, and resumes the same conversation after a clean or
provider-induced CLI exit. The controller owns cleanup and never relies on the
Agent's final text as the success signal.

`LIBERO_RUNTIME_DIR` is set by the schedulers to either
`libero/runtime/full` or `libero/runtime/strict`. Users normally invoke these
controllers through `agent_runner/run.py`.
