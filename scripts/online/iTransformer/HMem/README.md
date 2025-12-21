# H-Mem Experiment Scripts

This directory contains bash scripts for running H-Mem (Horizon-Bridging Neural Memory Network) experiments with iTransformer backbone.

## 📁 Files Overview

| Script | Purpose | Runtime |
|--------|---------|---------|
| `quick_test.sh` | Quick sanity check (1 iteration, horizon=24) | ~5-10 min |
| `ETTh2.sh` | Full experiments on ETTh2 dataset | ~2-3 hours |
| `ETTm1.sh` | Full experiments on ETTm1 dataset | ~2-3 hours |
| `Weather.sh` | Full experiments on Weather dataset | ~2-3 hours |
| `ECL.sh` | Full experiments on ECL dataset | ~2-3 hours |
| `Traffic.sh` | Full experiments on Traffic dataset | ~2-3 hours |
| `run_all.sh` | Run all datasets sequentially | ~10-15 hours |
| `ablation.sh` | Ablation study (test component contributions) | ~2-3 hours |
| `hyperparam_tuning.sh` | Hyperparameter search on ETTh2 | ~4-6 hours |

## 🚀 Quick Start

### 1. Quick Test (Recommended First Step)

Before running full experiments, verify your setup with a quick test:

```bash
cd /path/to/OnlineTSF-main
bash scripts/online/iTransformer/HMem/quick_test.sh
```

This will:
- Run H-Mem on ETTh2 with horizon=24
- Use small hyperparameters for speed
- Complete in ~5-10 minutes
- Output results to `logs/online/`

### 2. Single Dataset

To run experiments on a specific dataset:

```bash
# ETTh2 (Electricity Transformer - Hourly)
bash scripts/online/iTransformer/HMem/ETTh2.sh

# ETTm1 (Electricity Transformer - 15min)
bash scripts/online/iTransformer/HMem/ETTm1.sh

# Weather
bash scripts/online/iTransformer/HMem/Weather.sh

# ECL (Electricity Consuming Load)
bash scripts/online/iTransformer/HMem/ECL.sh

# Traffic
bash scripts/online/iTransformer/HMem/Traffic.sh
```

Each script runs 3 iterations on horizons [24, 48, 96].

### Baseline (P2+P7+P8)

Freeze the current best CHRC configuration (soft gating + horizon mask + buckets):

```bash
bash scripts/online/iTransformer/HMem/baseline_P2P7P8.sh ETTm1
bash scripts/online/iTransformer/HMem/baseline_P2P7P8.sh Weather
```

You can also pass a single horizon:

```bash
bash scripts/online/iTransformer/HMem/baseline_P2P7P8.sh ETTm1 96
```

### 3. All Datasets

To run H-Mem on all 5 datasets:

```bash
bash scripts/online/iTransformer/HMem/run_all.sh
```

**Warning**: This takes ~10-15 hours to complete!

## 🔬 Advanced Experiments

### Ablation Study

Test the contribution of different H-Mem components:

```bash
bash scripts/online/iTransformer/HMem/ablation.sh
```

This runs 4 configurations on ETTh2 (horizon=96):
1. **Full H-Mem**: SNMA + CHRC (both enabled)
2. **SNMA Only**: Neural memory without retrieval correction
3. **CHRC Only**: Retrieval correction without neural memory
4. **Baseline**: Frozen backbone, no adaptation

Results are saved to `logs/online/ablation/`.

### Hyperparameter Tuning

Search for optimal hyperparameters on ETTh2:

```bash
bash scripts/online/iTransformer/HMem/hyperparam_tuning.sh
```

Tunes 5 key parameters:
1. **Online Learning Rate**: [0.00001, 0.00003, 0.0001, 0.0003, 0.001]
2. **LoRA Rank**: [4, 8, 16, 32]
3. **Memory Dimension**: [128, 256, 512]
4. **POGT Ratio**: [0.25, 0.5, 0.75, 1.0]
5. **Memory Capacity**: [500, 1000, 2000, 5000]

Results are saved to `logs/online/hyperparam_tuning/`.

## ⚙️ Default Hyperparameters

```bash
# Model Settings
seq_len=336                 # Input sequence length
pred_len=[24, 48, 96]       # Prediction horizons

# LoRA Settings
lora_rank=8                 # LoRA rank
lora_alpha=16.0             # LoRA scaling factor
freeze=True                 # Freeze backbone weights

# Neural Memory (SNMA) Settings
memory_dim=256              # Memory state dimension
bottleneck_dim=32           # Bottleneck dimension
memory_momentum=0.9         # Memory update momentum
memory_num_heads=4          # Multi-head attention heads

# Error Memory Bank (CHRC) Settings
memory_capacity=1000        # Max entries in error bank
retrieval_top_k=5           # Retrieve top-k similar errors
chrc_feature_dim=128        # POGT encoding dimension
chrc_temperature=0.1        # Softmax temperature for aggregation
chrc_aggregation='softmax'  # Aggregation method

# POGT Settings
pogt_ratio=0.5              # POGT length = 0.5 * horizon

# Training Settings
hmem_warmup_steps=100       # SNMA warmup steps before joint training
learning_rate=0.0001        # Pretraining learning rate
online_learning_rate=0.0001 # Online learning rate
itr=3                       # Number of iterations
```

## 📊 Output

### Log Files

All logs are saved to `logs/online/` with naming format:
```
iTransformer_HMem_{dataset}_{horizon}_lora{rank}_mem{dim}_lr{lr}_onlinelr{olr}.log
```

Example:
```
iTransformer_HMem_ETTh2_96_lora8_mem256_lr0.0001_onlinelr0.0001.log
```

### Results Files

For each experiment, the following files are saved in `results/`:
```
results/{setting}/
├── metrics.npy     # MSE, MAE, RMSE, MAPE, MSPE
├── pred.npy        # Predictions [samples, horizon, features]
└── true.npy        # Ground truth [samples, horizon, features]
```

## 🛠️ Customization

### Modify Horizons

Edit the script to test different horizons:

```bash
# Original
for pred_len in 24 48 96
do
  ...
done

# Custom (e.g., long horizons)
for pred_len in 168 336 720
do
  ...
done
```

### Modify Learning Rates

```bash
# Original
for online_learning_rate in 0.0001
do
  ...
done

# Grid search
for online_learning_rate in 0.00001 0.0001 0.001
do
  ...
done
```

### Disable Components

Test without CHRC:
```bash
python run.py \
  ... \
  --use_snma True \
  --use_chrc False \  # Disable CHRC
  ...
```

Test without SNMA:
```bash
python run.py \
  ... \
  --use_snma False \  # Disable SNMA
  --use_chrc True \
  ...
```

## 📝 Notes

1. **GPU Memory**: H-Mem requires similar memory to the backbone model. For large datasets (Traffic, ECL), you may need to reduce batch size.

2. **Checkpoints**: By default, scripts use `--only_test` which loads pretrained checkpoints from `checkpoints/`. Make sure you have pretrained models before running.

3. **Iterations**: Scripts run 3 iterations (`--itr 3`) by default. Reduce to 1 for faster testing.

4. **Skip Existing**: Scripts use `--skip $filename` to avoid re-running completed experiments. Remove this flag to force re-run.

## 🔍 Monitoring Progress

### Real-time Log Viewing

```bash
# Watch log in real-time
tail -f logs/online/iTransformer_HMem_ETTh2_96_*.log
```

### Check Results

```bash
# View metrics
python -c "import numpy as np; print(np.load('results/{setting}/metrics.npy'))"
```

## 🐛 Troubleshooting

### Issue: "No module named 'exp.exp_hmem'"

**Solution**: Ensure H-Mem is registered in `exp/__init__.py`:
```python
def __getattr__(name):
    if name == 'Exp_HMem':
        from exp.exp_hmem import Exp_HMem
        return Exp_HMem
```

### Issue: CUDA out of memory

**Solutions**:
1. Reduce batch size: `--batch_size 8`
2. Reduce memory dimensions: `--memory_dim 128 --bottleneck_dim 16`
3. Reduce memory capacity: `--memory_capacity 500`

### Issue: Missing pretrained checkpoint

**Solution**: Either:
1. Pretrain the model first (remove `--only_test`)
2. Or download pretrained checkpoints

## 📚 References

For more details on H-Mem architecture and design, see:
- `adapter/hmem.py` - Main H-Mem module
- `exp/exp_hmem.py` - Experiment class with online learning logic
- `adapter/module/neural_memory.py` - SNMA implementation
- `util/error_bank.py` - CHRC implementation
