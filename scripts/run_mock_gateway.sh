#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
runtime_dir=${LIBERO_GATEWAY_RUNTIME_DIR:-$repo_root/.runtime/mock-gateway}
python_bin=${PYTHON:-python}
mkdir -p "$runtime_dir"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$repo_root/libero/src:$PYTHONPATH"
else
  export PYTHONPATH="$repo_root/libero/src"
fi

token=${LIBERO_API_TOKEN:-local-repro-token}
token_hash=$(TOKEN_VALUE="$token" "$python_bin" -c 'import hashlib,os; print(hashlib.sha256(os.environ["TOKEN_VALUE"].encode()).hexdigest())')
sed "s/__TOKEN_SHA256__/$token_hash/" \
  "$repo_root/libero/config/agents.example.json" >"$runtime_dir/agents.json"

export LIBERO_GATEWAY_BACKEND=mock
export LIBERO_GATEWAY_HOST=${LIBERO_GATEWAY_HOST:-127.0.0.1}
export LIBERO_GATEWAY_PORT=${LIBERO_GATEWAY_PORT:-18080}
export LIBERO_GATEWAY_AGENTS_FILE="$runtime_dir/agents.json"
export LIBERO_GATEWAY_AUDIT_LOG="$runtime_dir/audit.jsonl"
export LIBERO_GATEWAY_EVALUATIONS_DB="$runtime_dir/evaluations.sqlite3"
printf '%s\n' "mock_token=$token" "mock_url=http://$LIBERO_GATEWAY_HOST:$LIBERO_GATEWAY_PORT"
exec "$python_bin" -m libero_gateway.app
