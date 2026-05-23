#!/bin/bash

# This script is configured to train a small educational nanochat model
# (pretraining + finetuning) on a single RTX 3090 24GB GPU.
# It keeps the same overall pipeline as speedrun.sh, but scales down the
# model size, batch size, dataset size, and evaluation work.
#
# This is NOT expected to reach GPT-2 grade capability. Think of it as a
# complete small-model run that exercises tokenizer training, pretraining,
# SFT, checkpointing, evaluation, and chat on consumer hardware.

set -euo pipefail

# 1) Example launch (simplest):
# bash runs/run3090.sh
# 2) Example launch in a screen session (recommended for longer runs):
# screen -L -Logfile runs/run3090.log -S run3090 bash runs/run3090.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=run3090 screen -L -Logfile runs/run3090.log -S run3090 bash runs/run3090.sh
# 4) Example launch for a quick smoke test before the full run:
# MODEL_TAG=d16-test SKIP_TOKENIZER=1 NUM_ITERATIONS=100 SFT_NUM_ITERATIONS=100 TOTAL_BATCH_SIZE=131072 bash runs/run3090.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# 3090-sized knobs

# 中文说明：这些参数是 3090 完整小模型训练的主要调节入口；OOM 时优先降低 DEVICE_BATCH_SIZE。
MODEL_TAG="${MODEL_TAG:-d16-3090}"
DEPTH="${DEPTH:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-32}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
NUM_ITERATIONS="${NUM_ITERATIONS:--1}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-8}"
SFT_NUM_ITERATIONS="${SFT_NUM_ITERATIONS:-5000}"
SFT_LOAD_OPTIMIZER="${SFT_LOAD_OPTIMIZER:-0}"
SFT_EMBEDDING_LR="${SFT_EMBEDDING_LR:-0.02}"
SFT_UNEMBEDDING_LR="${SFT_UNEMBEDDING_LR:-0.0008}"
SFT_MATRIX_LR="${SFT_MATRIX_LR:-0.0015}"
EVAL_EVERY="${EVAL_EVERY:-100}"
SAMPLE_EVERY="${SAMPLE_EVERY:-100}"
EVAL_TOKENS="${EVAL_TOKENS:-524288}"
TOKENIZER_SHARDS="${TOKENIZER_SHARDS:-8}"
PRETRAIN_SHARDS="${PRETRAIN_SHARDS:-128}"
TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
RUN_CHAT_EVAL="${RUN_CHAT_EVAL:-0}"
SKIP_TOKENIZER="${SKIP_TOKENIZER:-0}"

echo "Running nanochat 3090 profile:"
echo "  MODEL_TAG=$MODEL_TAG"
echo "  DEPTH=$DEPTH"
echo "  MAX_SEQ_LEN=$MAX_SEQ_LEN"
echo "  DEVICE_BATCH_SIZE=$DEVICE_BATCH_SIZE"
echo "  TOTAL_BATCH_SIZE=$TOTAL_BATCH_SIZE"
echo "  NUM_ITERATIONS=$NUM_ITERATIONS"
echo "  TARGET_PARAM_DATA_RATIO=$TARGET_PARAM_DATA_RATIO"
echo "  SFT_NUM_ITERATIONS=$SFT_NUM_ITERATIONS"
echo "  SFT_LOAD_OPTIMIZER=$SFT_LOAD_OPTIMIZER"
echo "  SFT_EMBEDDING_LR=$SFT_EMBEDDING_LR"
echo "  SFT_UNEMBEDDING_LR=$SFT_UNEMBEDDING_LR"
echo "  SFT_MATRIX_LR=$SFT_MATRIX_LR"
echo "  TOKENIZER_SHARDS=$TOKENIZER_SHARDS"
echo "  PRETRAIN_SHARDS=$PRETRAIN_SHARDS"
echo "  TOKENIZER_MAX_CHARS=$TOKENIZER_MAX_CHARS"
echo "  SKIP_TOKENIZER=$SKIP_TOKENIZER"
echo "  NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"

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
#    `WANDB_RUN=run3090 bash runs/run3090.sh`
if [ -z "${WANDB_RUN:-}" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

if [ "$SKIP_TOKENIZER" = "1" ]; then
    # 中文说明：如果已经训练过 tokenizer，可以跳过这一段，只补齐 base_train 需要的数据 shard。
    python -m nanochat.dataset -n "$PRETRAIN_SHARDS"
else
    # Download the first ~2B characters of pretraining dataset
    # each data shard is ~250M chars
    # so we download 2e9 / 250e6 = 8 data shards at this point
    # each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
    # look at dev/repackage_data_reference.py for details on how this data was prepared
    python -m nanochat.dataset -n "$TOKENIZER_SHARDS"
    # Immediately also kick off downloading more shards in the background while tokenizer trains
    # Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
    # The maximum total number of shards available in the entire dataset is 6542.
    # For a 3090 learning run, a much smaller shard count is enough to exercise the pipeline.
    python -m nanochat.dataset -n "$PRETRAIN_SHARDS" &
    DATASET_DOWNLOAD_PID=$!
    # train the tokenizer with vocab size 2**15 = 32768 on up to ~2B characters of data
    python -m scripts.tok_train --max-chars="$TOKENIZER_MAX_CHARS"
    # evaluate the tokenizer (report compression ratio etc.)
    python -m scripts.tok_eval
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)
if [ "${DATASET_DOWNLOAD_PID:-}" != "" ]; then
    echo "Waiting for dataset download to complete..."
    wait "$DATASET_DOWNLOAD_PID"
fi

# d16 model by default: ~537M params, a larger 3090 profile that still follows
# speedrun's large-batch, target-param-data-ratio training horizon.
# 默认使用 target-param-data-ratio=8，让 base_train 像 speedrun 一样按模型参数量计算训练 token 数。
python -m scripts.base_train \
    --depth="$DEPTH" \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --core-metric-every=-1 \
    --sample-every="$SAMPLE_EVERY" \
    --num-iterations="$NUM_ITERATIONS" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"
# evaluate the model: BPB on train/val and a tiny CORE smoke test
python -m scripts.base_eval \
    --model-tag="$MODEL_TAG" \
    --device-batch-size=1 \
    --split-tokens=16384 \
    --max-per-task=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and optionally eval the model
python -m scripts.chat_sft \
    --model-tag="$MODEL_TAG" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --eval-every="$EVAL_EVERY" \
    --eval-tokens="$EVAL_TOKENS" \
    --chatcore-every=-1 \
    --num-iterations="$SFT_NUM_ITERATIONS" \
    --load-optimizer="$SFT_LOAD_OPTIMIZER" \
    --embedding-lr="$SFT_EMBEDDING_LR" \
    --unembedding-lr="$SFT_UNEMBEDDING_LR" \
    --matrix-lr="$SFT_MATRIX_LR" \
    --run="$WANDB_RUN"

if [ "$RUN_CHAT_EVAL" = "1" ]; then
    python -m scripts.chat_eval -i sft -g "$MODEL_TAG" -x 16 -b 1
fi

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -g d16-3090 -p "Hello, introduce yourself briefly."

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web -g d16-3090

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
