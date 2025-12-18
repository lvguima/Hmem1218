#!/bin/bash

# ============================================================================
# H-Mem: Ablation Study
# Tests different component combinations on ETTh2
# ============================================================================

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

if [ ! -d "./logs/online/ablation" ]; then
    mkdir ./logs/online/ablation
fi

# Basic settings
seq_len=336
data=ETTh2
model_name=iTransformer
pred_len=96  # Focus on longest horizon
learning_rate=0.0001
online_learning_rate=0.0001

# H-Mem hyperparameters
lora_rank=8
lora_alpha=16.0
memory_dim=256
bottleneck_dim=32
memory_capacity=1000
retrieval_top_k=5
pogt_ratio=0.5
hmem_warmup_steps=100

echo "=================================================="
echo "H-Mem Ablation Study on ETTh2"
echo "=================================================="
echo ""

# ============================================================================
# Ablation 1: Full H-Mem (SNMA + CHRC)
# ============================================================================
echo "[1/4] Running Full H-Mem (SNMA + CHRC)..."

online_method=HMem
filename=logs/online/ablation/$model_name'_HMem_Full_'$data'_'$pred_len.log

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

echo "Full H-Mem completed!"
echo ""

# ============================================================================
# Ablation 2: SNMA Only (No CHRC)
# ============================================================================
echo "[2/4] Running SNMA Only (w/o CHRC)..."

filename=logs/online/ablation/$model_name'_HMem_SNMA_only_'$data'_'$pred_len.log

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
  --use_chrc False \
  >> $filename 2>&1

echo "SNMA Only completed!"
echo ""

# ============================================================================
# Ablation 3: CHRC Only (No SNMA)
# ============================================================================
echo "[3/4] Running CHRC Only (w/o SNMA)..."

filename=logs/online/ablation/$model_name'_HMem_CHRC_only_'$data'_'$pred_len.log

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
  >> $filename 2>&1

echo "CHRC Only completed!"
echo ""

# ============================================================================
# Ablation 4: Baseline (No Adaptation)
# ============================================================================
echo "[4/4] Running Baseline (Frozen Backbone, No Adaptation)..."

filename=logs/online/ablation/$model_name'_HMem_Baseline_'$data'_'$pred_len.log

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
  --use_chrc False \
  >> $filename 2>&1

echo "Baseline completed!"
echo ""

echo "=================================================="
echo "Ablation Study Completed!"
echo ""
echo "Results Summary:"
echo "  1. Full H-Mem (SNMA + CHRC)"
echo "  2. SNMA Only (w/o CHRC)"
echo "  3. CHRC Only (w/o SNMA)"
echo "  4. Baseline (No Adaptation)"
echo ""
echo "Check logs/online/ablation/ for detailed results"
echo "=================================================="
