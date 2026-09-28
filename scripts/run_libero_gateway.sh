#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime_dir=${LIBERO_GATEWAY_RUNTIME_DIR:-$repo_root/.runtime/libero-gateway}
config_root=${LIBERO_CONFIG_PATH:-$repo_root/.runtime/libero-config}
python_bin=${PYTHON:-python}
mkdir -p "$runtime_dir"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$repo_root/libero/src:$PYTHONPATH"
else
  export PYTHONPATH="$repo_root/libero/src"
fi

if [ ! -f "$config_root/config.yaml" ]; then
  printf '%s\n' "LIBERO config is missing: $config_root/config.yaml" "run: make bootstrap" >&2
  exit 1
fi

token=$("$python_bin" -c 'import secrets; print("lbr_" + secrets.token_urlsafe(32))')
token_hash=$(TOKEN_VALUE="$token" "$python_bin" -c 'import hashlib,os; print(hashlib.sha256(os.environ["TOKEN_VALUE"].encode()).hexdigest())')
sed "s/__TOKEN_SHA256__/$token_hash/" \
  "$repo_root/libero/config/agents.example.json" >"$runtime_dir/agents.json"

export LIBERO_CONFIG_PATH="$config_root"
export LIBERO_GATEWAY_BACKEND=libero
export LIBERO_GATEWAY_HOST=${LIBERO_GATEWAY_HOST:-127.0.0.1}
export LIBERO_GATEWAY_PORT=${LIBERO_GATEWAY_PORT:-18080}
export LIBERO_GATEWAY_GPU_IDS=${LIBERO_GATEWAY_GPU_IDS:-0}
export LIBERO_GATEWAY_MAX_SESSIONS_PER_GPU=${LIBERO_GATEWAY_MAX_SESSIONS_PER_GPU:-1}
export LIBERO_GATEWAY_AGENTS_FILE="$runtime_dir/agents.json"
export LIBERO_GATEWAY_AUDIT_LOG="$runtime_dir/audit.jsonl"
export LIBERO_GATEWAY_EVALUATIONS_DB="$runtime_dir/evaluations.sqlite3"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

printf '%s\n' "gateway=http://$LIBERO_GATEWAY_HOST:$LIBERO_GATEWAY_PORT" "gpus=$LIBERO_GATEWAY_GPU_IDS"
exec "$python_bin" -m libero_gateway.app
