#!/bin/bash
set -eo pipefail

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is currently set up for a d12 letter-counting / SpellingBee sweep.

# 1) Example launch (simplest):
# bash runs/exp2.sh
# 2) Quick test run (100 iterations for base_train and chat_sft):
# bash runs/exp2.sh --testrun
# 3) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/exp2.log -S exp2 bash runs/exp2.sh
# 4) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=exp2 screen -L -Logfile runs/exp2.log -S exp2 bash runs/exp2.sh

# -----------------------------------------------------------------------------
# User-configurable parameters
DEPTH=12
SPELLINGBEE_SIZES=(5000 20000 40000 80000)
SPELLINGBEE_VAL_SIZE=2000
CHAT_EVAL_TASKS="SpellingBee-Val"
CHAT_EVAL_MAX_PROBLEMS="--max-problems=300"

# -----------------------------------------------------------------------------
# Shared exp2 config (e.g., NANOCHAT_BASE_DIR_RAW)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/exp2_env.sh" ]]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/exp2_env.sh"
fi

# -----------------------------------------------------------------------------
# Parse command line arguments
TESTRUN=false
for arg in "$@"; do
    case $arg in
        --testrun)
            TESTRUN=true
            shift
            ;;
    esac
done

if [ "$TESTRUN" = true ]; then
    echo "TESTRUN mode: using depth=12, --num-iterations=100 for base_train and chat_sft, --max-problems=16 for chat_eval"
    BASE_TRAIN_HORIZON="--num-iterations=100"
    SFT_HORIZON="--num-iterations=100"
    CHAT_EVAL_MAX_PROBLEMS="--max-problems=16"
else
    BASE_TRAIN_HORIZON="--target-param-data-ratio=8.5"
    SFT_HORIZON=""
    CHAT_EVAL_MAX_PROBLEMS="--max-problems=300"
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
# for Lambda instance, only files under ~/instance-name are kept.
if [[ -z "${NANOCHAT_BASE_DIR_RAW:-}" ]]; then
    NANOCHAT_BASE_DIR_RAW='~/nanochat-exp2/.cache/nanochat'
fi
NANOCHAT_BASE_DIR="$(eval echo "$NANOCHAT_BASE_DIR_RAW")"
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Run identity (date + commit hash)
RUN_DATE="$(date +%Y-%m-%d)"
GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_ID="${RUN_DATE}_${GIT_HASH}"
export RUN_DATE GIT_HASH RUN_ID
export NANOCHAT_RUN_ID="$RUN_ID"
RUN_DIR="$NANOCHAT_BASE_DIR/runs/$RUN_ID"
export NANOCHAT_RUN_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR"
mkdir -p "$NANOCHAT_BASE_DIR/runs"
echo "$RUN_ID" > "$NANOCHAT_BASE_DIR/runs/latest"

# -----------------------------------------------------------------------------
# GPU detection: auto-detect number and type of GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}  # default to 1 if nvidia-smi fails
LAUNCHER="torchrun --standalone --nproc_per_node=$NUM_GPUS"

GPU_NAME=$(nvidia-smi -L 2>/dev/null | head -1)
if echo "$GPU_NAME" | grep -qi "H100"; then
    echo "Detected H100: enabling --fp8 and default window pattern"
    GPU_FLAGS="--fp8"
elif echo "$GPU_NAME" | grep -qi "A100"; then
    echo "Detected A100: no --fp8, using --window-pattern L (SDPA lacks sliding window)"
    GPU_FLAGS="--window-pattern L"
else
    echo "Unknown GPU ($GPU_NAME): defaulting to A100 settings (no --fp8, --window-pattern L)"
    GPU_FLAGS="--window-pattern L"
fi

echo "Detected $NUM_GPUS GPU(s), using: $LAUNCHER"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_logged() {
    local log_file="$1"
    shift
    "$@" 2>&1 | tee "$log_file"
}

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d12 bash runs/exp2.sh`
if [ -z "${WANDB_RUN:-}" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

wandb_run_name() {
    local tag="$1"
    if [ "$WANDB_RUN" = "dummy" ]; then
        printf 'dummy'
    else
        printf '%s_%s' "$WANDB_RUN" "$tag"
    fi
}

echo "---------------------"
echo "BEGINNING NEW RUN"
echo "num GPUs: $NUM_GPUS. testrun=$TESTRUN. depth=$DEPTH."
echo "base dir: $NANOCHAT_BASE_DIR"
echo "run dir: $RUN_DIR"
echo "run id: $RUN_ID"
echo "---------------------"

RESULTS_DIR="$RUN_DIR/lettercount_sweep"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/models.csv"
if [ ! -f "$RESULTS_FILE" ]; then
    echo "stage,spellingbee_size,model_tag,log_file" > "$RESULTS_FILE"
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

echo "---------------------"
echo "Beginning dataset download and tokenization"
echo "---------------------"
# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

echo "---------------------"
echo "Beginning pretraining"
echo "---------------------"

BASE_TAG="exp2_${RUN_ID}_d${DEPTH}_base"
BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_TAG"
if ls "$BASE_CKPT_DIR"/model_*.pt 1>/dev/null 2>&1; then
    echo "Base model checkpoint found at $BASE_CKPT_DIR, skipping base_train and base_eval"
else
    BASE_TRAIN_LOG="$RESULTS_DIR/${BASE_TAG}_train.log"
    run_logged "$BASE_TRAIN_LOG" $LAUNCHER -m scripts.base_train -- --depth=$DEPTH $BASE_TRAIN_HORIZON --device-batch-size=16 --core-metric-every=-1 $GPU_FLAGS --run="$(wandb_run_name "$BASE_TAG")" --model-tag="$BASE_TAG"
    echo "base,0,$BASE_TAG,$BASE_TRAIN_LOG" >> "$RESULTS_FILE"

    echo "---------------------"
    echo "beginning post-pretrain eval"
    echo "---------------------"

    # evaluate the model: BPB on train/val and draw samples (skip CORE; it is too slow)
    BASE_EVAL_LOG="$RESULTS_DIR/${BASE_TAG}_base_eval.log"
    run_logged "$BASE_EVAL_LOG" $LAUNCHER -m scripts.base_eval -- --eval bpb,sample --device-batch-size=16 --model-tag="$BASE_TAG"
fi

echo "---------------------"
echo "beginning base chat eval"
echo "---------------------"
BASE_CHAT_EVAL_LOG="$RESULTS_DIR/${BASE_TAG}_chat_eval.log"
run_logged "$BASE_CHAT_EVAL_LOG" $LAUNCHER -m scripts.chat_eval -- -i base -g "$BASE_TAG" -a "$CHAT_EVAL_TASKS" --spellingbee-val-size="$SPELLINGBEE_VAL_SIZE" $CHAT_EVAL_MAX_PROBLEMS

# -----------------------------------------------------------------------------
# SFT sweep (vary SpellingBee size while keeping the same base checkpoint)
echo "---------------------"
echo "Beginning SFT sweep"
echo "---------------------"

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

for SPELLINGBEE_SIZE in "${SPELLINGBEE_SIZES[@]}"; do
    SFT_TAG="exp2_${RUN_ID}_d${DEPTH}_sb${SPELLINGBEE_SIZE}"
    SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/$SFT_TAG"
    if ls "$SFT_CKPT_DIR"/model_*.pt 1>/dev/null 2>&1; then
        echo "SFT checkpoint found at $SFT_CKPT_DIR, skipping chat_sft"
    else
        SFT_TRAIN_LOG="$RESULTS_DIR/${SFT_TAG}_train.log"
        run_logged "$SFT_TRAIN_LOG" $LAUNCHER -m scripts.chat_sft -- $SFT_HORIZON --device-batch-size=16 --run="$(wandb_run_name "$SFT_TAG")" --base-model-tag="$BASE_TAG" --model-tag="$SFT_TAG" --spellingbee-size="$SPELLINGBEE_SIZE" --spellingbee-val-size="$SPELLINGBEE_VAL_SIZE"
        echo "sft,$SPELLINGBEE_SIZE,$SFT_TAG,$SFT_TRAIN_LOG" >> "$RESULTS_FILE"
    fi

    echo "---------------------"
    echo "beginning post-SFT eval for $SFT_TAG"
    echo "---------------------"

    SFT_EVAL_LOG="$RESULTS_DIR/${SFT_TAG}_chat_eval.log"
    run_logged "$SFT_EVAL_LOG" $LAUNCHER -m scripts.chat_eval -- -i sft -g "$SFT_TAG" -a "$CHAT_EVAL_TASKS" --spellingbee-val-size="$SPELLINGBEE_VAL_SIZE" $CHAT_EVAL_MAX_PROBLEMS
done

# -----------------------------------------------------------------------------
# RL (reinforcement learning on GSM8K)
# Temporarily commented out while exp2 is focused on the d12 SpellingBee sweep.
# echo "---------------------"
# echo "Beginning RL"
# echo "---------------------"
# $LAUNCHER -m scripts.chat_rl -- --device-batch-size=8 --run=$WANDB_RUN
# echo "---------------------"
# echo "beginning post-RL eval"
# echo "---------------------"
# $LAUNCHER -m scripts.chat_eval -- -i rl $CHAT_EVAL_MAX_PROBLEMS

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
echo ""
echo "Report generation"
echo ""
python -m nanochat.report generate
