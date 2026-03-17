#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $(basename "$0") <ip> [user] [remote_repo_dir]"
  echo "Env overrides: REMOTE_BASE_DIR, REMOTE_REPO_DIR, DEST_DIR, REMOTE_USER"
  exit 1
fi

IP="$1"
REMOTE_USER="${2:-${REMOTE_USER:-ubuntu}}"
REMOTE_REPO_DIR="${3:-${REMOTE_REPO_DIR:-${REMOTE_REPO_DIR_RAW:-}}}"
REMOTE="${REMOTE_USER}@${IP}"
SSH_OPTS="-o StrictHostKeyChecking=accept-new"

# Shared exp2 config (e.g., NANOCHAT_BASE_DIR_RAW)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/exp2_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/exp2_env.sh"
fi

# exp2.sh sets: export NANOCHAT_BASE_DIR=\"$HOME/nanochat-exp2/.cache/nanochat\"
REMOTE_BASE_DIR_RAW="${REMOTE_BASE_DIR:-${NANOCHAT_BASE_DIR_RAW:-~/nanochat-exp2/.cache/nanochat}}"
REMOTE_BASE_DIR_EXPANDED="$(ssh $SSH_OPTS "$REMOTE" "echo $REMOTE_BASE_DIR_RAW")"
if [[ -z "$REMOTE_BASE_DIR_EXPANDED" ]]; then
  echo "Failed to resolve remote base dir (REMOTE_BASE_DIR=$REMOTE_BASE_DIR_RAW)."
  exit 1
fi

# Find repo root on remote if not provided.
if [[ -z "$REMOTE_REPO_DIR" ]]; then
  echo "Locating repo on remote (looking for runs/exp2.sh)..."
  REMOTE_EXP2_PATH="$(ssh $SSH_OPTS "$REMOTE" "find ~ -maxdepth 6 -type f -path '*/runs/exp2.sh' -print -quit")"
  if [[ -z "$REMOTE_EXP2_PATH" ]]; then
    echo "Could not find runs/exp2.sh on remote. Pass remote repo dir as 3rd arg or set REMOTE_REPO_DIR."
    exit 1
  fi
  REMOTE_REPO_DIR="$(dirname "$(dirname "$REMOTE_EXP2_PATH")")"
fi

DEST_DIR="${DEST_DIR:-./runs/exp2_logs/${IP}}"
mkdir -p "$DEST_DIR/logs" "$DEST_DIR/runs"

copy_file_if_exists() {
  local remote_path="$1"
  local dest_dir="$2"
  if ssh $SSH_OPTS "$REMOTE" "test -f \"$remote_path\""; then
    rsync -a -e "ssh $SSH_OPTS" "$REMOTE:$remote_path" "$dest_dir/"
    echo "Copied file: $remote_path"
  else
    echo "Missing file on remote: $remote_path"
  fi
}

copy_dir_if_exists() {
  local remote_dir="$1"
  local dest_dir="$2"
  if ssh $SSH_OPTS "$REMOTE" "test -d \"$remote_dir\""; then
    mkdir -p "$dest_dir"
    rsync -a -e "ssh $SSH_OPTS" "$REMOTE:$remote_dir/" "$dest_dir/"
    echo "Copied dir: $remote_dir"
  else
    echo "Missing dir on remote: $remote_dir"
  fi
}

# 1) Screen/stdout log (if run with: screen -L -Logfile runs/exp2.log ...)
copy_file_if_exists "$REMOTE_REPO_DIR/runs/exp2.log" "$DEST_DIR/logs"

# 2) Sync all runs (rsync will skip unchanged files)
copy_dir_if_exists "$REMOTE_BASE_DIR_EXPANDED/runs" "$DEST_DIR/runs"

echo "Done. Files copied to: $DEST_DIR"
