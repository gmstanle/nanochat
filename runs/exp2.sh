#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

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
DEPTH=26

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
    DEPTH=12
    BASE_TRAIN_HORIZON="--num-iterations=100"
    SFT_HORIZON="--num-iterations=100"
    CHAT_EVAL_MAX_PROBLEMS="--max-problems=16"
else
    BASE_TRAIN_HORIZON="--target-param-data-ratio=8.5"
    SFT_HORIZON=""
    CHAT_EVAL_MAX_PROBLEMS=""
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
# for Lambda instance, only files under ~/instance-name are kept.
export NANOCHAT_BASE_DIR="$HOME/nanochat-exp2/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

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
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

echo "---------------------"
echo "BEGINNING NEW RUN"
echo "num GPUs: $NUM_GPUS. testrun=$TESTRUN. depth=$DEPTH."
echo "dir for run checkpoints and eval reports: $NANOCHAT_BASE_DIR"
echo "---------------------"

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

BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d$DEPTH"
if ls "$BASE_CKPT_DIR"/model_*.pt 1>/dev/null 2>&1; then
    echo "Base model checkpoint found at $BASE_CKPT_DIR, skipping base_train and base_eval"
else
    $LAUNCHER -m scripts.base_train -- --depth=$DEPTH $BASE_TRAIN_HORIZON --device-batch-size=16 $GPU_FLAGS --run=$WANDB_RUN
    
    echo "---------------------"
    echo "beginning post-pretrain eval"
    echo "---------------------"

    # evaluate the model: CORE metric, BPB on train/val, and draw samples
    $LAUNCHER -m scripts.base_eval -- --device-batch-size=16
fi

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)
echo "---------------------"
echo "Beginning SFT"
echo "---------------------"

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

SFT_CKPT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d$DEPTH"
if ls "$SFT_CKPT_DIR"/model_*.pt 1>/dev/null 2>&1; then
    echo "SFT checkpoint found at $SFT_CKPT_DIR, skipping chat_sft and chat_eval"
else
    # run SFT and eval the model
    $LAUNCHER -m scripts.chat_sft -- $SFT_HORIZON --device-batch-size=16 --run=$WANDB_RUN

    echo "---------------------"
    echo "beginning post-SFT eval"
    echo "---------------------"

    $LAUNCHER -m scripts.chat_eval -- -i sft $CHAT_EVAL_MAX_PROBLEMS
fi

# -----------------------------------------------------------------------------
# RL (reinforcement learning on GSM8K)

echo "---------------------"
echo "Beginning RL"
echo "---------------------"

# Note: no --num-iterations available; uses --num-epochs (default 1 = ~466 steps)
$LAUNCHER -m scripts.chat_rl -- --device-batch-size=8 --run=$WANDB_RUN
echo "---------------------"
echo "beginning post-RL eval"
echo "---------------------"

$LAUNCHER -m scripts.chat_eval -- -i rl $CHAT_EVAL_MAX_PROBLEMS

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
