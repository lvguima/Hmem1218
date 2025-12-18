#!/bin/bash

# ============================================================================
# H-Mem: Quick Test
# Fast sanity check to verify H-Mem implementation works
# Uses single iteration and short horizon
# ============================================================================

# Initialize conda if available
CONDA_INIT_SCRIPT=""
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT_SCRIPT="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT_SCRIPT="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/mnt/d/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT_SCRIPT="/mnt/d/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/d/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT_SCRIPT="/d/anaconda3/etc/profile.d/conda.sh"
elif [ -f "D:/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_INIT_SCRIPT="D:/anaconda3/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_INIT_SCRIPT" ]; then
    source "$CONDA_INIT_SCRIPT"
    echo "Initialized conda from: $CONDA_INIT_SCRIPT"
fi

# Try to activate conda environment 'cl' if conda is available
if command -v conda &> /dev/null; then
    # Check if cl environment exists
    if conda env list 2>/dev/null | grep -q "cl"; then
        echo "Activating conda environment: cl"
        conda activate cl 2>/dev/null || source activate cl 2>/dev/null || true
        if [ -n "$CONDA_PREFIX" ]; then
            echo "Conda environment activated: $CONDA_PREFIX"
        fi
    fi
fi

# Detect Python command (prefer conda environment Python)
PYTHON_CMD=""
if [ -n "$CONDA_PREFIX" ]; then
    # Use conda environment Python
    if [ -f "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_CMD="$CONDA_PREFIX/bin/python"
    elif [ -f "$CONDA_PREFIX/bin/python3" ]; then
        PYTHON_CMD="$CONDA_PREFIX/bin/python3"
    fi
fi

# Fallback: Try direct paths for cl environment
if [ -z "$PYTHON_CMD" ]; then
    if [ -f "/mnt/d/anaconda3/envs/cl/bin/python" ]; then
        PYTHON_CMD="/mnt/d/anaconda3/envs/cl/bin/python"
    elif [ -f "/mnt/d/anaconda3/envs/cl/bin/python3" ]; then
        PYTHON_CMD="/mnt/d/anaconda3/envs/cl/bin/python3"
    elif [ -f "/d/anaconda3/envs/cl/bin/python" ]; then
        PYTHON_CMD="/d/anaconda3/envs/cl/bin/python"
    elif [ -f "D:/anaconda3/envs/cl/python.exe" ]; then
        PYTHON_CMD="D:/anaconda3/envs/cl/python.exe"
    elif [ -f "/d/anaconda3/envs/cl/python.exe" ]; then
        PYTHON_CMD="/d/anaconda3/envs/cl/python.exe"
    fi
fi

# Final fallback: system Python
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    elif command -v python &> /dev/null; then
        PYTHON_CMD=python
    elif command -v py &> /dev/null; then
        PYTHON_CMD=py
    else
        echo "Error: Python not found. Please install Python or add it to PATH."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD"
# Verify numpy is available
if $PYTHON_CMD -c "import numpy; print('numpy version:', numpy.__version__)" 2>/dev/null; then
    echo "✓ numpy is available"
else
    echo "⚠ Warning: numpy not found in this Python environment"
    echo "  Please ensure you're using the correct conda environment"
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/online" ]; then
    mkdir ./logs/online
fi

# Basic settings
seq_len=96
data=ETTh2
model_name=iTransformer
online_method=HMem
pred_len=24  # Shortest horizon for quick test

# H-Mem hyperparameters (smaller for speed)
lora_rank=4
lora_alpha=8.0
memory_dim=128
bottleneck_dim=16
memory_capacity=100
retrieval_top_k=3
pogt_ratio=0.5
hmem_warmup_steps=10

learning_rate=0.0001
online_learning_rate=0.0001

echo "=================================================="
echo "H-Mem Quick Test"
echo "=================================================="
echo ""
echo "Dataset: $data"
echo "Model: $model_name"
echo "Horizon: $pred_len"
echo "Iterations: 1 (quick test)"
echo ""
echo "Running..."
echo ""

filename=logs/online/${model_name}_${online_method}_quick_test.log

$PYTHON_CMD -u run.py \
  --dataset $data --border_type 'online' \
  --model $model_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --online_method $online_method \
  --itr 1 \
  --only_test \
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
  2>&1 | tee $filename

echo ""
echo "=================================================="
echo "Quick test completed!"
echo "Log saved to: $filename"
echo "=================================================="
