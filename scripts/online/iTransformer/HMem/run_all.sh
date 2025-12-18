#!/bin/bash

# ============================================================================
# H-Mem: Run All Experiments
# This script runs H-Mem on all 5 datasets with iTransformer backbone
# ============================================================================

echo "=================================================="
echo "H-Mem Experiments - Running All Datasets"
echo "=================================================="
echo ""
echo "Datasets: ETTh2, ETTm1, Weather, ECL, Traffic"
echo "Model: iTransformer"
echo "Horizons: 24, 48, 96"
echo "Iterations: 3"
echo ""
echo "=================================================="
echo ""

# Make scripts executable
chmod +x scripts/online/iTransformer/HMem/*.sh

# Run experiments sequentially
echo "[1/5] Running ETTh2 experiments..."
bash scripts/online/iTransformer/HMem/ETTh2.sh
echo ""

echo "[2/5] Running ETTm1 experiments..."
bash scripts/online/iTransformer/HMem/ETTm1.sh
echo ""

echo "[3/5] Running Weather experiments..."
bash scripts/online/iTransformer/HMem/Weather.sh
echo ""

echo "[4/5] Running ECL experiments..."
bash scripts/online/iTransformer/HMem/ECL.sh
echo ""

echo "[5/5] Running Traffic experiments..."
bash scripts/online/iTransformer/HMem/Traffic.sh
echo ""

echo "=================================================="
echo "All H-Mem experiments completed!"
echo "Results saved in: logs/online/"
echo "=================================================="
