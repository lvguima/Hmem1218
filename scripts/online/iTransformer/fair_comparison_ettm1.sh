#!/bin/bash

# Fair comparison script for OnlineTSF-main to match FSM Titan-Stream experiment
# Matches parameters from FSM/scripts/online_forecast/titan_ettm1_delayed.sh
#
# Key matched parameters:
# - Dataset: ETTm1 with online split (20/5/75 in OnlineTSF ≈ 4/1/15 months in FSM)
# - Seq_len: 512, Pred_len: 96
# - Model architecture: d_model=256, n_heads=4, e_layers=2, d_ff=512
# - Training: batch_size=32, learning_rate=5e-4 (pretraining)
# - Online: Multiple methods for comparison

set -e

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

# Dataset and model configuration
DATA=ETTm1
MODEL=iTransformer
SEQ_LEN=512
PRED_LEN=96
LABEL_LEN=48

# Model architecture (matched to FSM Titan-Stream)
D_MODEL=256
N_HEADS=4
E_LAYERS=2
D_FF=512

# Training hyperparameters (matched to FSM)
PRETRAIN_LR=0.0005  # 5e-4
BATCH_SIZE=32
TRAIN_EPOCHS=15
PATIENCE=3

# Online learning rates (tuned for different methods)
# Note: OnlineTSF typically uses much smaller online LR than pretraining LR
ONLINE_LR_NAIVE=0.000003      # Naive online gradient descent
ONLINE_LR_PROCEED=0.00001     # PROCEED method
ONLINE_LR_SOLID=0.00001       # SOLID method
ONLINE_LR_ONENET=0.00001      # OneNet method

# PROCEED-specific parameters
CONCEPT_DIM=200
BOTTLENECK_DIM=32
TUNE_MODE=down_up

# Number of iterations for statistical significance
ITR=3

echo "=========================================="
echo "Fair Comparison Experiment: ETTm1"
echo "Matching FSM Titan-Stream Configuration"
echo "=========================================="
echo "Dataset: $DATA (online split: 20/5/75)"
echo "Seq_len: $SEQ_LEN, Pred_len: $PRED_LEN"
echo "Model: $MODEL (d_model=$D_MODEL, n_heads=$N_HEADS, e_layers=$E_LAYERS)"
echo "Pretraining: lr=$PRETRAIN_LR, batch_size=$BATCH_SIZE, epochs=$TRAIN_EPOCHS"
echo "Iterations: $ITR"
echo "=========================================="

# ============================================
# Step 1: Pretraining (if not already done)
# ============================================
echo ""
echo "[1/5] Pretraining $MODEL on $DATA..."
echo "--------------------------------------"

PRETRAIN_LOG=logs/pretrain_${MODEL}_${DATA}_${PRED_LEN}.log

python -u run.py \
  --dataset $DATA \
  --border_type 'online' \
  --model $MODEL \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --learning_rate $PRETRAIN_LR \
  --lradj type3 \
  --train_epochs $TRAIN_EPOCHS \
  --patience $PATIENCE \
  --itr $ITR \
  --save_opt \
  > $PRETRAIN_LOG 2>&1

echo "Pretraining completed. Log: $PRETRAIN_LOG"

# ============================================
# Step 2: Static Baseline (No Online Learning)
# ============================================
echo ""
echo "[2/5] Testing Static Baseline (no adaptation)..."
echo "--------------------------------------"

STATIC_LOG=logs/online/${MODEL}_Static_${DATA}_${PRED_LEN}.log

python -u run.py \
  --dataset $DATA \
  --border_type 'online' \
  --model $MODEL \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --itr $ITR \
  --only_test \
  --save_opt \
  --skip $STATIC_LOG \
  > $STATIC_LOG 2>&1

echo "Static baseline completed. Log: $STATIC_LOG"

# ============================================
# Step 3: Naive Online Learning
# ============================================
echo ""
echo "[3/5] Testing Naive Online Learning..."
echo "--------------------------------------"

ONLINE_LOG=logs/online/${MODEL}_Online_${DATA}_${PRED_LEN}_onlinelr${ONLINE_LR_NAIVE}.log

python -u run.py \
  --dataset $DATA \
  --border_type 'online' \
  --model $MODEL \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_ff $D_FF \
  --batch_size $BATCH_SIZE \
  --online_method Online \
  --online_learning_rate $ONLINE_LR_NAIVE \
  --itr $ITR \
  --only_test \
  --save_opt \
  --skip $ONLINE_LOG \
  --pin_gpu True \
  --reduce_bs False \
  > $ONLINE_LOG 2>&1

echo "Naive online learning completed. Log: $ONLINE_LOG"

# ============================================
# Step 4: PROCEED Method (Main Contribution)
# ============================================
echo ""
echo "[4/5] Testing PROCEED Method..."
echo "--------------------------------------"

PROCEED_LOG=logs/online/${MODEL}_Proceed_${DATA}_${PRED_LEN}_mid${CONCEPT_DIM}_btl${BOTTLENECK_DIM}_onlinelr${ONLINE_LR_PROCEED}.log

python -u run.py \
  --dataset $DATA \
  --border_type 'online' \
  --model $MODEL \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_ff $D_FF \
  --batch_size 16 \
  --online_method Proceed \
  --tune_mode $TUNE_MODE \
  --concept_dim $CONCEPT_DIM \
  --bottleneck_dim $BOTTLENECK_DIM \
  --online_learning_rate $ONLINE_LR_PROCEED \
  --learning_rate $PRETRAIN_LR \
  --lradj type3 \
  --val_online_lr \
  --diff_online_lr \
  --itr $ITR \
  --only_test \
  --pretrain \
  --save_opt \
  --skip $PROCEED_LOG \
  > $PROCEED_LOG 2>&1

echo "PROCEED method completed. Log: $PROCEED_LOG"

# ============================================
# Step 5: SOLID Method (KDD 2024 Baseline)
# ============================================
echo ""
echo "[5/5] Testing SOLID Method..."
echo "--------------------------------------"

SOLID_LOG=logs/online/${MODEL}_SOLID_${DATA}_${PRED_LEN}_onlinelr${ONLINE_LR_SOLID}.log

python -u run.py \
  --dataset $DATA \
  --border_type 'online' \
  --model $MODEL \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --d_model $D_MODEL \
  --n_heads $N_HEADS \
  --e_layers $E_LAYERS \
  --d_ff $D_FF \
  --batch_size 16 \
  --online_method SOLID \
  --online_learning_rate $ONLINE_LR_SOLID \
  --itr $ITR \
  --only_test \
  --save_opt \
  --skip $SOLID_LOG \
  > $SOLID_LOG 2>&1

echo "SOLID method completed. Log: $SOLID_LOG"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results saved in logs/online/"
echo ""
echo "To compare with FSM Titan-Stream results:"
echo "1. Check MSE, MAE metrics in log files"
echo "2. Compare online adaptation speed"
echo "3. Analyze computational efficiency"
echo ""
echo "Log files:"
echo "  - Pretraining: $PRETRAIN_LOG"
echo "  - Static: $STATIC_LOG"
echo "  - Online: $ONLINE_LOG"
echo "  - PROCEED: $PROCEED_LOG"
echo "  - SOLID: $SOLID_LOG"
echo "=========================================="
