#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <ip> [user] [remote_repo_dir] [run_id]"
  echo "Env overrides: REMOTE_BASE_DIR, REMOTE_REPO_DIR, DEST_DIR, REMOTE_USER, RUN_ID"
  exit 1
fi

IP="$1"
REMOTE_USER="${2:-${REMOTE_USER:-$USER}}"
REMOTE_REPO_DIR="${3:-${REMOTE_REPO_DIR:-}}"
RUN_ID="${4:-${RUN_ID:-}}"
REMOTE="${REMOTE_USER}@${IP}"

# Shared exp2 config (e.g., NANOCHAT_BASE_DIR_RAW)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/exp2_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/exp2_env.sh"
fi

# exp2.sh sets: export NANOCHAT_BASE_DIR=\"$HOME/nanochat-exp2/.cache/nanochat\"
REMOTE_BASE_DIR_RAW="${REMOTE_BASE_DIR:-${NANOCHAT_BASE_DIR_RAW:-~/nanochat-exp2/.cache/nanochat}}"
REMOTE_BASE_DIR_EXPANDED="$(ssh "$REMOTE" "echo $REMOTE_BASE_DIR_RAW")"
if [[ -z "$REMOTE_BASE_DIR_EXPANDED" ]]; then
  echo "Failed to resolve remote base dir (REMOTE_BASE_DIR=$REMOTE_BASE_DIR_RAW)."
  exit 1
fi

# Determine run id (prefer explicit, else latest on remote)
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(ssh "$REMOTE" "cat \"$REMOTE_BASE_DIR_EXPANDED/runs/latest\" 2>/dev/null")"
fi
if [[ -n "$RUN_ID" ]]; then
  REMOTE_RUN_DIR="$REMOTE_BASE_DIR_EXPANDED/runs/$RUN_ID"
  REMOTE_REPORT_DIR="$REMOTE_RUN_DIR/report"
  REMOTE_REPORT_MD="$REMOTE_REPORT_DIR/report.md"
else
  REMOTE_RUN_DIR=""
  REMOTE_REPORT_DIR="$REMOTE_BASE_DIR_EXPANDED/report"
  REMOTE_REPORT_MD="$REMOTE_REPORT_DIR/report.md"
fi

# Find repo root on remote if not provided.
if [[ -z "$REMOTE_REPO_DIR" ]]; then
  echo "Locating repo on remote (looking for runs/exp2.sh)..."
  REMOTE_EXP2_PATH="$(ssh "$REMOTE" "find ~ -maxdepth 6 -type f -path '*/runs/exp2.sh' -print -quit")"
  if [[ -z "$REMOTE_EXP2_PATH" ]]; then
    echo "Could not find runs/exp2.sh on remote. Pass remote repo dir as 3rd arg or set REMOTE_REPO_DIR."
    exit 1
  fi
  REMOTE_REPO_DIR="$(dirname "$(dirname "$REMOTE_EXP2_PATH")")"
fi

# Use report creation time (remote) for destination folder timestamp if possible.
REMOTE_REPORT_TS="$(ssh "$REMOTE" "if [ -f \"$REMOTE_REPORT_MD\" ]; then \
  if stat --version >/dev/null 2>&1; then stat -c %Y \"$REMOTE_REPORT_MD\"; else stat -f %m \"$REMOTE_REPORT_MD\"; fi; \
elif [ -d \"$REMOTE_REPORT_DIR\" ]; then \
  NEWEST_FILE=\$(ls -1t \"$REMOTE_REPORT_DIR\" 2>/dev/null | head -1); \
  if [ -n \"\$NEWEST_FILE\" ]; then \
    if stat --version >/dev/null 2>&1; then stat -c %Y \"$REMOTE_REPORT_DIR/\$NEWEST_FILE\"; else stat -f %m \"$REMOTE_REPORT_DIR/\$NEWEST_FILE\"; fi; \
  fi; \
fi")"

if [[ -n "${REMOTE_REPORT_TS:-}" ]]; then
  if date -r 0 >/dev/null 2>&1; then
    REMOTE_REPORT_STAMP="$(date -r "$REMOTE_REPORT_TS" +%Y%m%d_%H%M%S)"
  else
    REMOTE_REPORT_STAMP="$(date -d "@$REMOTE_REPORT_TS" +%Y%m%d_%H%M%S)"
  fi
else
  REMOTE_REPORT_STAMP="$(date +%Y%m%d_%H%M%S)"
fi

DEST_DIR="${DEST_DIR:-./runs/exp2_logs/${IP}_${REMOTE_REPORT_STAMP}}"
mkdir -p "$DEST_DIR/logs" "$DEST_DIR/report"

copy_file_if_exists() {
  local remote_path="$1"
  local dest_dir="$2"
  if ssh "$REMOTE" "test -f \"$remote_path\""; then
    rsync -a "$REMOTE:$remote_path" "$dest_dir/"
    echo "Copied file: $remote_path"
  else
    echo "Missing file on remote: $remote_path"
  fi
}

copy_dir_if_exists() {
  local remote_dir="$1"
  local dest_dir="$2"
  if ssh "$REMOTE" "test -d \"$remote_dir\""; then
    rsync -a "$REMOTE:$remote_dir/" "$dest_dir/"
    echo "Copied dir: $remote_dir"
  else
    echo "Missing dir on remote: $remote_dir"
  fi
}

# 1) Screen/stdout log (if run with: screen -L -Logfile runs/exp2.log ...)
copy_file_if_exists "$REMOTE_REPO_DIR/runs/exp2.log" "$DEST_DIR/logs"

# 2) Report sections + aggregated report
copy_dir_if_exists "$REMOTE_REPORT_DIR" "$DEST_DIR/report"

# 3) Convenience report.md copied to repo root by nanochat.report generate
copy_file_if_exists "$REMOTE_REPO_DIR/report.md" "$DEST_DIR/report"

# 4) Run config if available
if [[ -n "$REMOTE_RUN_DIR" ]]; then
  copy_file_if_exists "$REMOTE_RUN_DIR/config.json" "$DEST_DIR/report"
fi

echo "Done. Files copied to: $DEST_DIR"
