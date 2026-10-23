#!/bin/bash

# Showing an example run for exercising some of the code paths on 2 x 5060 Ti 16GB GPU

# Run as:
# bash runs/dual5060ti.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your 5060 Ti.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
# uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# 1. Download ~35 shards (~3.5GB, enough for ~1.7B tokens so d12 doesn't repeat data)
uv run python -m nanochat.dataset -n 35
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# 2. Pretrain GPT-1 sized d12 model on 2 x 5060 Ti 16GB
# Auto-computes: head_dim=128, max_seq_len=2048, total_batch_size=524288, num_iterations=2520 (~1.32B tokens)
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=12 \
    --attn-impl=fa2 \
    --fp8 \
    --window-pattern=L \
    --device-batch-size=8 \
    --eval-every=250 \
    --eval-tokens=5242880 \
    --core-metric-every=-1 \
    --core-metric-max-per-task=-1 \
    --sample-every=500 \
    --save-every=500 \
    --run=$WANDB_RUN

# 3. Full CORE + BPB evaluation on both GPUs
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.base_eval -- \
    --device-batch-size=8

# # 4. SFT on both GPUs
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
    --device-batch-size=8 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --chatcore-every=999999 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -p "What is the capital of France?"
