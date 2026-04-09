#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLUSTER_NAME="nanochat-exp2-d12"
TASK_YAML="$REPO_ROOT/sky/nanochat-exp2-full.yaml"
SKY_ENV_BIN="/Users/geoffstanley/.venvs/skypilot/bin"
SKY_BIN="$SKY_ENV_BIN/sky"

export AWS_PROFILE=skypilot
export PATH="$SKY_ENV_BIN:$PATH"

cd "$REPO_ROOT"
"$SKY_BIN" check
"$SKY_BIN" launch -y -d -c "$CLUSTER_NAME" -i 10 "$TASK_YAML"
