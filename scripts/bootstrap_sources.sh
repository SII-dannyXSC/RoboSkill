#!/usr/bin/env sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

bootstrap_source() {
  label=$1
  upstream_url=$2
  base_commit=$3
  expected_tree=$4
  target=$5
  patch_dir=$6

  if [ -e "$target" ]; then
    if [ ! -d "$target/.git" ]; then
      printf '%s\n' "$label target exists but is not a Git checkout: $target" >&2
      exit 1
    fi
    if [ -n "$(git -C "$target" status --porcelain)" ]; then
      printf '%s\n' "$label checkout has local changes: $target" >&2
      exit 1
    fi
  else
    mkdir -p "$(dirname -- "$target")"
    git clone "$upstream_url" "$target"
    git -C "$target" switch --create robot-repro-patched "$base_commit"
    git -C "$target" am "$patch_dir"/*.patch
  fi

  actual_tree=$(git -C "$target" rev-parse 'HEAD^{tree}')
  if [ "$actual_tree" != "$expected_tree" ]; then
    printf '%s\n' \
      "$label tree mismatch at $target" \
      "expected=$expected_tree" \
      "actual=$actual_tree" >&2
    exit 1
  fi
  printf '%s\n' "$label=$target" "$label.tree=$actual_tree"
}

bootstrap_source \
  libero \
  "${LIBERO_UPSTREAM_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}" \
  8f1084e3132a39270c3a13ebe37270a43ece2a01 \
  8b1c5e7faed6c486cecc9453be8b23e4cd0d25f5 \
  "${LIBERO_SOURCE_DIR:-$repo_root/third_party/LIBERO}" \
  "$repo_root/libero/patches"
