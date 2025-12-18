#!/bin/bash

# ============================================================================
# H-Mem: Horizon-Bridging Neural Memory Network
# Dataset: Weather
# Model: iTransformer
# ============================================================================

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

# Basic settings
seq_len=336
data=Weather
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

# ============================================================================
# Experiment: Standard horizons (24, 48, 96)
# ============================================================================

for pred_len in 24 48 96
do
for learning_rate in 0.0001
do
for online_learning_rate in 0.0001
do
  suffix='_lr'$learning_rate'_onlinelr'$online_learning_rate
  filename=logs/online/$model_name'_'$online_method'_'$data'_'$pred_len'_lora'$lora_rank'_mem'$memory_dim$suffix.log

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
    --use_snma True \
    --use_chrc True \
    >> $filename 2>&1
done
done
done

echo "Weather experiments completed!"
