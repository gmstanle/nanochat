#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

echo "=========================================="
echo "EXPERIMENT 1: SpellingBee Mistake Recovery"
echo "Start time: $(date)"
echo "=========================================="
echo ""
EXPERIMENT_START=$(date +%s)

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/nanochat-exp1/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

RESULTS_DIR="$NANOCHAT_BASE_DIR/spellingbee_results"
mkdir -p $RESULTS_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

echo "=========================================="
echo "Starting Python venv setup"
echo "Start time: $(date)"
SETUP_START=$(date +%s)

# install uv (if not already installed)
echo "Installing uv if needed..."
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
echo "Creating venv if needed..."
[ -d ".venv" ] || uv venv
# install the repo dependencies
echo "Syncing dependencies..."
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
echo "Activating venv..."
source .venv/bin/activate

SETUP_END=$(date +%s)
SETUP_TIME=$((SETUP_END - SETUP_START))
echo "Completed Python venv setup in ${SETUP_TIME}s"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.

echo "=========================================="
echo "Initializing report system"
echo "=========================================="
python -m nanochat.report reset
echo "Report system initialized"
echo ""

# -----------------------------------------------------------------------------
# Tokenizer

echo "=========================================="
echo "Starting Tokenizer Training"
echo "Start time: $(date)"
TOK_START=$(date +%s)

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
echo "Downloading 8 data shards for tokenizer training..."
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 370 is the right number here
echo "Starting background download of 370 shards..."
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
echo "Training tokenizer (vocab_size=65536, max_chars=2B)..."
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=65536
# evaluate the tokenizer (report compression ratio etc.)
echo "Evaluating tokenizer..."
python -m scripts.tok_eval

TOK_END=$(date +%s)
TOK_TIME=$((TOK_END - TOK_START))
echo "Completed Tokenizer Training in ${TOK_TIME}s"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "=========================================="
echo "Starting Base Model Pretraining"
echo "Start time: $(date)"
BASE_START=$(date +%s)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. Also, the new DataLoader wastes about 35% of tokens to cropping
# so 240 / (1 - 0.35) = 370 shards are needed.
# At ~100MB/shard, this downloads ~37GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID
echo "Dataset download complete"

# Number of processes/GPUs to use
NPROC_PER_NODE=8

# pretrain the d20 model
echo "Pretraining d20 model (561M params, Chinchilla 20x)..."
PRETRAIN_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --target-param-data-ratio=20 --run=$WANDB_RUN
PRETRAIN_END=$(date +%s)
PRETRAIN_TIME=$((PRETRAIN_END - PRETRAIN_START))
echo "Completed pretraining in ${PRETRAIN_TIME}s"

# evaluate the model on a larger chunk of train/val data and draw some samples
echo "Evaluating base loss on train/val splits..."
BASELOSS_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
BASELOSS_END=$(date +%s)
BASELOSS_TIME=$((BASELOSS_END - BASELOSS_START))
echo "Completed base_loss evaluation in ${BASELOSS_TIME}s"

# evaluate the model on CORE tasks
echo "Evaluating on CORE benchmark tasks..."
BASEEVAL_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval
BASEEVAL_END=$(date +%s)
BASEEVAL_TIME=$((BASEEVAL_END - BASEEVAL_START))
echo "Completed CORE evaluation in ${BASEEVAL_TIME}s"

BASE_END=$(date +%s)
BASE_TIME=$((BASE_END - BASE_START))
echo "Completed Base Model Pretraining (total) in ${BASE_TIME}s"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

echo "=========================================="
echo "Starting Midtraining"
echo "Start time: $(date)"
MID_START=$(date +%s)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
echo "Downloading synthetic identity conversations..."
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
echo "Running midtraining (conversation format, tool use, multiple choice)..."
MIDTRAIN_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
MIDTRAIN_END=$(date +%s)
MIDTRAIN_TIME=$((MIDTRAIN_END - MIDTRAIN_START))
echo "Completed midtraining in ${MIDTRAIN_TIME}s"

echo "Evaluating midtrained model..."
MIDEVAL_START=$(date +%s)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid
MIDEVAL_END=$(date +%s)
MIDEVAL_TIME=$((MIDEVAL_END - MIDEVAL_START))
echo "Completed mid evaluation in ${MIDEVAL_TIME}s"

MID_END=$(date +%s)
MID_TIME=$((MID_END - MID_START))
echo "Completed Midtraining (total) in ${MID_TIME}s"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

echo "=========================================="
echo "Starting SFT Experiments (Mistakes Loop)"
echo "Start time: $(date)"
SFT_OVERALL_START=$(date +%s)

# train sft and re-eval right away (should see a small bump)
# Unique tag for this run
MISTAKES=("True" "False")
for m in "${MISTAKES[@]}"; do

    echo "=========================================="
    echo "Starting SFT run with mistakes=${m}"
    echo "Start time: $(date)"
    echo "=========================================="

    TAG="mistakes_${m}"

    # Record start time
    START_TIME=$(date +%s)

    # Train the model with fixed flops budget
    # The script will auto-calculate num_iterations to hit target_flops
    # CORE eval happens once at the end (999999 ensures only final step)
    echo "Training SFT model (mistakes=${m})..."
    SFT_TRAIN_START=$(date +%s)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
        --depth=$d \
        --run="${WANDB_RUN}_${TAG}" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --use-mistakes="${m}" \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"
    SFT_TRAIN_END=$(date +%s)
    SFT_TRAIN_TIME=$((SFT_TRAIN_END - SFT_TRAIN_START))
    echo "Completed SFT training (mistakes=${m}) in ${SFT_TRAIN_TIME}s"

    echo "Evaluating SFT model (mistakes=${m})..."
    SFT_EVAL_START=$(date +%s)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft --model-tag="${TAG}"
    SFT_EVAL_END=$(date +%s)
    SFT_EVAL_TIME=$((SFT_EVAL_END - SFT_EVAL_START))
    echo "Completed SFT evaluation (mistakes=${m}) in ${SFT_EVAL_TIME}s"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    echo "=========================================="
    echo "Completed SFT run with mistakes=${m}"
    echo "Total time: ${TRAIN_TIME}s (train: ${SFT_TRAIN_TIME}s, eval: ${SFT_EVAL_TIME}s)"
    echo "End time: $(date)"
    echo "=========================================="
    echo ""

done

SFT_OVERALL_END=$(date +%s)
SFT_OVERALL_TIME=$((SFT_OVERALL_END - SFT_OVERALL_START))
echo "Completed All SFT Experiments in ${SFT_OVERALL_TIME}s"
echo "=========================================="
echo ""

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience

echo "=========================================="
echo "Generating Final Report"
echo "Start time: $(date)"
REPORT_START=$(date +%s)

python -m nanochat.report generate

REPORT_END=$(date +%s)
REPORT_TIME=$((REPORT_END - REPORT_START))
echo "Completed report generation in ${REPORT_TIME}s"
echo "=========================================="
echo ""

EXPERIMENT_END=$(date +%s)
EXPERIMENT_TIME=$((EXPERIMENT_END - EXPERIMENT_START))
EXPERIMENT_HOURS=$((EXPERIMENT_TIME / 3600))
EXPERIMENT_MINS=$(((EXPERIMENT_TIME % 3600) / 60))
EXPERIMENT_SECS=$((EXPERIMENT_TIME % 60))

echo "=========================================="
echo "EXPERIMENT COMPLETE"
echo "End time: $(date)"
echo "Total experiment time: ${EXPERIMENT_HOURS}h ${EXPERIMENT_MINS}m ${EXPERIMENT_SECS}s (${EXPERIMENT_TIME}s)"
echo "=========================================="
