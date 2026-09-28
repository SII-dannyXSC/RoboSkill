#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$repo_root"

forbidden_names=$(find . \( \
  -path './.git' -o -path './.venv*' -o -path '*/node_modules' -o \
  -path './third_party' -o \
  -path './.runtime' -o -path './runs' -o -path './batches' -o \
  -path './sessions' -o -path './artifacts' -o -path './videos' -o \
  -path '*/__pycache__' -o -path '*.egg-info' \
  \) -prune -o -type f \( \
  -name '.env' -o -name 'agents.json' -o -name '*.sqlite' -o \
  -name '*.sqlite3' -o -name '*.jsonl' -o -name '*.log' -o \
  -name '*.pem' -o -name '*.key' -o -name '*.hdf5' -o \
  -name '*.h5' -o -name '*.mp4' \) -print)
if [ -n "$forbidden_names" ]; then
  printf '%s\n' 'Forbidden runtime or credential files:' "$forbidden_names" >&2
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  tracked_runtime=$(git ls-files --cached --others --exclude-standard | \
    grep -E '(^|/)(\.env|agent\.local\.json|agents\.json|\.runtime|runs|batches|sessions|artifacts|videos|skills|plugins|demos|orchestration|ops)(/|$)' | \
    while IFS= read -r path; do [ ! -e "$path" ] || printf '%s\n' "$path"; done || true)
  if [ -n "$tracked_runtime" ]; then
    printf '%s\n' 'Tracked runtime, credential, or generated-skill paths:' "$tracked_runtime" >&2
    exit 1
  fi
fi

for removed_boundary in plugins demos orchestration ops patches; do
  if [ -e "$removed_boundary" ]; then
    printf '%s\n' "Obsolete public boundary still exists: $removed_boundary" >&2
    exit 1
  fi
done
if find . -type f \( -name 'SKILL.md' -o -path '*/.codex-plugin/*' -o -path '*/.claude-plugin/*' \) \
  -not -path './.git/*' -print -quit | grep -q .; then
  printf '%s\n' 'Generated skill or plugin metadata is present.' >&2
  exit 1
fi

patches=$(find libero/patches -maxdepth 1 -type f -name '*.patch' -print)
if [ "$(printf '%s\n' "$patches" | sed '/^$/d' | wc -l | tr -d ' ')" != 1 ] || \
   ! printf '%s\n' "$patches" | grep -q 'camera-only'; then
  printf '%s\n' 'libero/patches must contain only the camera-only patch.' >&2
  exit 1
fi

findings=$(mktemp "${TMPDIR:-/tmp}/agent-for-robot-portability.XXXXXX")
trap 'rm -f "$findings"' EXIT HUP INT TERM
if grep -RIlE 'JD_API_KEY|JD_KIMI|jd_kimi_health|jd-kimi|/Users/[^/]+|/Volumes/[^/]+|/data/xiesicheng|/home/xiesicheng|com\.xiesicheng|api-main\.gacc\.cc|api\.macc\.cc|(^|[^[:alnum:]_])sk-[A-Za-z0-9_-]{20,}' \
  --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.venv312 \
  --exclude-dir=node_modules --exclude-dir=__pycache__ --exclude-dir='*.egg-info' \
  --exclude-dir=.runtime --exclude-dir=runs --exclude-dir=batches \
  --exclude-dir=sessions --exclude-dir=artifacts --exclude-dir=videos \
  --exclude='verify_publication.sh' . >"$findings"; then
  printf '%s\n' 'Non-portable path or possible secret found:' >&2
  sed -n '1,120p' "$findings" >&2
  exit 1
fi

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git diff --check
fi
printf '%s\n' 'publication_check=ok'
