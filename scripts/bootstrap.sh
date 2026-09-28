#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
conda_bin=${CONDA:-conda}
env_name=${LIBERO_ENV_NAME:-agent-for-robot-libero}

if ! command -v "$conda_bin" >/dev/null 2>&1; then
  printf '%s\n' 'Conda is required to create the pinned LIBERO/MuJoCo environment.' >&2
  exit 1
fi

"$repo_root/scripts/bootstrap_sources.sh"
if "$conda_bin" run -n "$env_name" python -c 'import bddl, mujoco, numpy, robosuite' >/dev/null 2>&1; then
  printf '%s\n' "reusing_conda_env=$env_name"
elif "$conda_bin" run -n "$env_name" python -c 'import sys' >/dev/null 2>&1; then
  "$conda_bin" env update -n "$env_name" -f "$repo_root/libero/env.yml" --prune
else
  "$conda_bin" env create -n "$env_name" -f "$repo_root/libero/env.yml"
fi
"$conda_bin" run -n "$env_name" python -m pip install -e "$repo_root/third_party/LIBERO"
"$conda_bin" run -n "$env_name" python -m pip install -e "$repo_root/libero[test]"
"$conda_bin" run -n "$env_name" python "$repo_root/scripts/configure_libero.py" \
  --source "$repo_root/third_party/LIBERO" \
  --config-root "$repo_root/.runtime/libero-config"

printf '%s\n' \
  "conda_env=$env_name" \
  "libero_source=$repo_root/third_party/LIBERO" \
  "next=conda activate $env_name" \
  "then=install and authenticate Codex CLI or Claude Code, then copy an agent_runner config"
