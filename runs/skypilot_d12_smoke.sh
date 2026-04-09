#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLUSTER_NAME="nanochat-d12-smoke"
TASK_YAML="$REPO_ROOT/sky/nanochat-d12-smoke.yaml"
SKY_BIN="/Users/geoffstanley/.venvs/skypilot/bin/sky"
SKY_ENV_BIN="/Users/geoffstanley/.venvs/skypilot/bin"

export AWS_PROFILE=skypilot
export PATH="$SKY_ENV_BIN:$PATH"

cd "$REPO_ROOT"
"$SKY_BIN" check
"$SKY_BIN" launch -c "$CLUSTER_NAME" -i 10 "$TASK_YAML"
