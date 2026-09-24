#!/usr/bin/env bash

fastafd_repo_root() {
  local start="${1:-.}"
  local dir

  if [[ -f "$start" ]]; then
    dir="$(cd "$(dirname "$start")" && pwd)"
  else
    dir="$(cd "$start" && pwd)"
  fi

  local git_root
  if git_root="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null)"; then
    printf '%s\n' "$git_root"
    return 0
  fi

  while [[ "$dir" != "/" ]]; do
    if [[ -d "$dir/scripts" && -d "$dir/python" ]]; then
      printf '%s\n' "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done

  echo "Failed to locate FastAFD repository root from: $start" >&2
  return 1
}

fastafd_init_paths() {
  REPO_ROOT="$(fastafd_repo_root "${1:-.}")"
  CALIB_DIR="$REPO_ROOT"
  SCRIPT_ROOT="$REPO_ROOT/scripts"
  export REPO_ROOT CALIB_DIR SCRIPT_ROOT
}
