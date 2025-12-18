# Fair comparison script for OnlineTSF-main to match FSM Titan-Stream experiment
# PowerShell version for Windows
# Matches parameters from FSM/scripts/online_forecast/titan_ettm1_delayed.sh

$ErrorActionPreference = "Stop"

# Create log directories
if (-not (Test-Path "./logs")) {
    New-Item -ItemType Directory -Path "./logs" | Out-Null
}

if (-not (Test-Path "./logs/online")) {
    New-Item -ItemType Directory -Path "./logs/online" | Out-Null
}

# Dataset and model configuration
$DATA = "ETTm1"
$MODEL = "iTransformer"
$SEQ_LEN = 512
$PRED_LEN = 96
$LABEL_LEN = 48

# Model architecture (matched to FSM Titan-Stream)
$D_MODEL = 256
$N_HEADS = 4
$E_LAYERS = 2
$D_FF = 512

# Training hyperparameters (matched to FSM)
$PRETRAIN_LR = 0.0005  # 5e-4
$BATCH_SIZE = 32
$TRAIN_EPOCHS = 15
$PATIENCE = 3

# Online learning rates
$ONLINE_LR_NAIVE = 0.000003
$ONLINE_LR_PROCEED = 0.00001
$ONLINE_LR_SOLID = 0.00001

# PROCEED-specific parameters
$CONCEPT_DIM = 200
$BOTTLENECK_DIM = 32
$TUNE_MODE = "down_up"

# Number of iterations
$ITR = 3

Write-Host "=========================================="
Write-Host "Fair Comparison Experiment: ETTm1"
Write-Host "Matching FSM Titan-Stream Configuration"
Write-Host "=========================================="
Write-Host "Dataset: $DATA (online split: 20/5/75)"
Write-Host "Seq_len: $SEQ_LEN, Pred_len: $PRED_LEN"
Write-Host "Model: $MODEL (d_model=$D_MODEL, n_heads=$N_HEADS, e_layers=$E_LAYERS)"
Write-Host "Pretraining: lr=$PRETRAIN_LR, batch_size=$BATCH_SIZE, epochs=$TRAIN_EPOCHS"
Write-Host "Iterations: $ITR"
Write-Host "=========================================="

# ============================================
# Step 1: Pretraining
# ============================================
Write-Host ""
Write-Host "[1/5] Pretraining $MODEL on $DATA..."
Write-Host "--------------------------------------"

$PRETRAIN_LOG = "logs/pretrain_${MODEL}_${DATA}_${PRED_LEN}.log"

python -u run.py `
  --dataset $DATA `
  --border_type 'online' `
  --model $MODEL `
  --seq_len $SEQ_LEN `
  --label_len $LABEL_LEN `
  --pred_len $PRED_LEN `
  --d_model $D_MODEL `
  --n_heads $N_HEADS `
  --e_layers $E_LAYERS `
  --d_ff $D_FF `
  --batch_size $BATCH_SIZE `
  --learning_rate $PRETRAIN_LR `
  --lradj type3 `
  --train_epochs $TRAIN_EPOCHS `
  --patience $PATIENCE `
  --itr $ITR `
  --save_opt `
  > $PRETRAIN_LOG 2>&1

Write-Host "Pretraining completed. Log: $PRETRAIN_LOG"

# ============================================
# Step 2: Static Baseline
# ============================================
Write-Host ""
Write-Host "[2/5] Testing Static Baseline (no adaptation)..."
Write-Host "--------------------------------------"

$STATIC_LOG = "logs/online/${MODEL}_Static_${DATA}_${PRED_LEN}.log"

python -u run.py `
  --dataset $DATA `
  --border_type 'online' `
  --model $MODEL `
  --seq_len $SEQ_LEN `
  --label_len $LABEL_LEN `
  --pred_len $PRED_LEN `
  --d_model $D_MODEL `
  --n_heads $N_HEADS `
  --e_layers $E_LAYERS `
  --d_ff $D_FF `
  --batch_size $BATCH_SIZE `
  --itr $ITR `
  --only_test `
  --save_opt `
  --skip $STATIC_LOG `
  > $STATIC_LOG 2>&1

Write-Host "Static baseline completed. Log: $STATIC_LOG"

# ============================================
# Step 3: Naive Online Learning
# ============================================
Write-Host ""
Write-Host "[3/5] Testing Naive Online Learning..."
Write-Host "--------------------------------------"

$ONLINE_LOG = "logs/online/${MODEL}_Online_${DATA}_${PRED_LEN}_onlinelr${ONLINE_LR_NAIVE}.log"

python -u run.py `
  --dataset $DATA `
  --border_type 'online' `
  --model $MODEL `
  --seq_len $SEQ_LEN `
  --label_len $LABEL_LEN `
  --pred_len $PRED_LEN `
  --d_model $D_MODEL `
  --n_heads $N_HEADS `
  --e_layers $E_LAYERS `
  --d_ff $D_FF `
  --batch_size $BATCH_SIZE `
  --online_method Online `
  --online_learning_rate $ONLINE_LR_NAIVE `
  --itr $ITR `
  --only_test `
  --save_opt `
  --skip $ONLINE_LOG `
  --pin_gpu True `
  --reduce_bs False `
  > $ONLINE_LOG 2>&1

Write-Host "Naive online learning completed. Log: $ONLINE_LOG"

# ============================================
# Step 4: PROCEED Method
# ============================================
Write-Host ""
Write-Host "[4/5] Testing PROCEED Method..."
Write-Host "--------------------------------------"

$PROCEED_LOG = "logs/online/${MODEL}_Proceed_${DATA}_${PRED_LEN}_mid${CONCEPT_DIM}_btl${BOTTLENECK_DIM}_onlinelr${ONLINE_LR_PROCEED}.log"

python -u run.py `
  --dataset $DATA `
  --border_type 'online' `
  --model $MODEL `
  --seq_len $SEQ_LEN `
  --label_len $LABEL_LEN `
  --pred_len $PRED_LEN `
  --d_model $D_MODEL `
  --n_heads $N_HEADS `
  --e_layers $E_LAYERS `
  --d_ff $D_FF `
  --batch_size 16 `
  --online_method Proceed `
  --tune_mode $TUNE_MODE `
  --concept_dim $CONCEPT_DIM `
  --bottleneck_dim $BOTTLENECK_DIM `
  --online_learning_rate $ONLINE_LR_PROCEED `
  --learning_rate $PRETRAIN_LR `
  --lradj type3 `
  --val_online_lr `
  --diff_online_lr `
  --itr $ITR `
  --only_test `
  --pretrain `
  --save_opt `
  --skip $PROCEED_LOG `
  > $PROCEED_LOG 2>&1

Write-Host "PROCEED method completed. Log: $PROCEED_LOG"

# ============================================
# Step 5: SOLID Method
# ============================================
Write-Host ""
Write-Host "[5/5] Testing SOLID Method..."
Write-Host "--------------------------------------"

$SOLID_LOG = "logs/online/${MODEL}_SOLID_${DATA}_${PRED_LEN}_onlinelr${ONLINE_LR_SOLID}.log"

python -u run.py `
  --dataset $DATA `
  --border_type 'online' `
  --model $MODEL `
  --seq_len $SEQ_LEN `
  --label_len $LABEL_LEN `
  --pred_len $PRED_LEN `
  --d_model $D_MODEL `
  --n_heads $N_HEADS `
  --e_layers $E_LAYERS `
  --d_ff $D_FF `
  --batch_size 16 `
  --online_method SOLID `
  --online_learning_rate $ONLINE_LR_SOLID `
  --itr $ITR `
  --only_test `
  --save_opt `
  --skip $SOLID_LOG `
  > $SOLID_LOG 2>&1

Write-Host "SOLID method completed. Log: $SOLID_LOG"

# ============================================
# Summary
# ============================================
Write-Host ""
Write-Host "=========================================="
Write-Host "All experiments completed!"
Write-Host "=========================================="
Write-Host "Results saved in logs/online/"
Write-Host ""
Write-Host "Log files:"
Write-Host "  - Pretraining: $PRETRAIN_LOG"
Write-Host "  - Static: $STATIC_LOG"
Write-Host "  - Online: $ONLINE_LOG"
Write-Host "  - PROCEED: $PROCEED_LOG"
Write-Host "  - SOLID: $SOLID_LOG"
Write-Host "=========================================="
