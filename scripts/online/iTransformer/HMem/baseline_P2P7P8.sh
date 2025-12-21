#!/bin/bash

# ============================================================================
# H-Mem Baseline: P2 + P7 + P8 (softmax, bucket=4)
# Usage:
#   bash scripts/online/iTransformer/HMem/baseline_P2P7P8.sh [DATASET] [PRED_LEN]
# Example:
#   bash scripts/online/iTransformer/HMem/baseline_P2P7P8.sh ETTm1 96
# ============================================================================

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

if [ ! -d "./logs/online/baseline" ]; then
    mkdir ./logs/online/baseline
fi

# Basic settings
seq_len=336
data=${1:-ETTm1}
model_name=iTransformer
online_method=HMem

# H-Mem hyperparameters
lora_rank=8
lora_alpha=16.0
memory_dim=256
bottleneck_dim=32
memory_capacity=1000
retrieval_top_k=5
pogt_ratio=0.5
hmem_warmup_steps=100

learning_rate=0.0001
online_learning_rate=0.0001

pred_lens=(24 48 96)
if [ -n "$2" ]; then
    pred_lens=("$2")
fi

for pred_len in "${pred_lens[@]}"
do
  suffix='_lr'$learning_rate'_onlinelr'$online_learning_rate
  filename=logs/online/baseline/$model_name'_'$online_method'_'$data'_'$pred_len'_P2P7P8'$suffix.log

  python -u run.py \
    --dataset $data --border_type 'online' \
    --model $model_name \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --online_method $online_method \
    --itr 3 --skip $filename \
    --save_opt --only_test \
    --learning_rate $learning_rate \
    --online_learning_rate $online_learning_rate \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --memory_dim $memory_dim \
    --bottleneck_dim $bottleneck_dim \
    --memory_capacity $memory_capacity \
    --retrieval_top_k $retrieval_top_k \
    --pogt_ratio $pogt_ratio \
    --hmem_warmup_steps $hmem_warmup_steps \
    --freeze True \
    --use_snma False \
    --use_chrc True \
    --chrc_trust_threshold 0.5 \
    --chrc_gate_steepness 10.0 \
    --chrc_use_horizon_mask True \
    --chrc_horizon_mask_mode exp \
    --chrc_horizon_mask_decay 0.98 \
    --chrc_horizon_mask_min 0.2 \
    --chrc_use_buckets True \
    --chrc_bucket_num 4 \
    --chrc_aggregation softmax \
    >> $filename 2>&1
done

echo "Baseline P2+P7+P8 completed for $data!"
