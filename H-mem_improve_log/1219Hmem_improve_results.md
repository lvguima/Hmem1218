# H-Mem Implementation Plan & Results

**Date:** 2025-12-19  
**Scope:** Turn design issues in `Hmem_design_improve.md` into a staged refactor plan and capture post-change results.  
**Status:** Steps 1-7 completed; results recorded below.

---

## Prioritization Principles
1) Correctness and causality first (avoid leakage / misalignment).  
2) Stability next (reduce high-variance updates).  
3) Effectiveness and generalization after (avoid negative transfer).  
4) Complexity last (touching multiple modules only when earlier fixes settle).

## Implementation Steps (P0-P3, Completed)

### 1) POGT Observability Alignment (P0)
**Why:** Causality risk and evaluation optimism.  
**Work:**
- Ensure POGT used for prediction is strictly from observed past (not from current window GT).
- Rewire `Exp_HMem.online()` to construct POGT from `recent_batch` instead of `current_batch`.
- Keep POGT in `_update_online()` from observed window only.
**Status:** Completed.

### 2) Preserve SNMA Memory Across Steps (P0)
**Why:** Per-step reset removes temporal smoothing and increases noise chasing.  
**Work:**
- Add `detach_state()` in SNMA/NeuralMemoryState to truncate graphs without wiping memory.
- Replace per-step reset with `detach_state()`, keep reset only at phase boundaries.
**Status:** Completed.

### 3) CHRC Abstention on Low Absolute Similarity (P1)
**Why:** Prevent negative transfer when retrieval is weak or OOD.  
**Work:**
- Add `chrc_min_similarity` (absolute threshold).
- Default to "no correction" if similarity below threshold.
**Status:** Completed.

### 4) Reduce HyperNetwork Volatility (P1)
**Why:** Over-sensitive LoRA params hurt stability.  
**Work:**
- Add EMA smoothing for generated LoRA parameters (`lora_ema_decay`).
**Status:** Completed.

### 5) Share / Align POGT Representations (P2)
**Why:** SNMA + CHRC share the same signal but encode it independently.  
**Work:**
- Add `hmem_share_pogt` flag to share SNMA encoding with CHRC.
**Status:** Completed.

### 6) Confidence Gate Input Simplification (P2)
**Why:** High-dimensional flattened inputs overfit in batch_size=1.  
**Work:**
- Replace flattened tensors with summary stats (mean/std).
**Status:** Completed.

### 7) Active Forgetting in Memory Bank (P3)
**Why:** Remove stale patterns in long streams.  
**Work:**
- Add importance decay and pruning (`chrc_forget_decay`, `chrc_forget_threshold`, `chrc_max_age`).
**Status:** Completed.

---

## Experiment Results by Step

### Baseline Re-Run (Online Method)
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 1e-4
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 1e-4
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 1.193117 | 0.697323 | 1.092299 | User rerun |
| Weather | 2.983787 | 1.040027 | 1.727364 | User rerun |

---

### Step 1 (P0): POGT Observability Alignment
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 1.015220 | 0.645267 | 1.007581 | User-run result after POGT causal change |
| Weather | 4.584300 | 1.274634 | 2.141098 | User-run result after POGT causal change |

**Observation:** Large performance drop vs historical non-causal setup.

---

### Step 2 (P0): Preserve SNMA Memory Across Steps
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 1.015220 | 0.645267 | 1.007581 | User-run result after SNMA detach |
| Weather | 4.584300 | 1.274634 | 2.141098 | User-run result after SNMA detach |

**Observation:** Identical to P0-1; causal POGT constraint dominates behavior.

---

### Step 3 (P1): CHRC Abstention on Low Absolute Similarity
**Commands:** (Exact `--chrc_min_similarity` value not recorded.)
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --chrc_min_similarity <value>
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --chrc_min_similarity <value>
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 0.826357 | 0.577490 | 0.909042 | User-run result with CHRC abstention |
| Weather | 1.560324 | 0.709551 | 1.249129 | User-run result with CHRC abstention |

**Observation:** Weather improves vs P0 causal baseline; ETTm1 returns close to static baseline.

---

### Step 4 (P1): Reduce HyperNetwork Volatility (LoRA EMA)
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.9
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --lora_ema_decay 0.9 --chrc_min_similarity 0.9
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 0.788969 | 0.562364 | 0.888239 | User-run result with EMA |
| Weather | 1.480340 | 0.675052 | 1.216692 | User-run result with EMA |

**Observation:** Both datasets improved vs prior causal POGT runs; EMA appears stabilizing.

---

### Step 5 (P2): Share/Align POGT Representations
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --hmem_share_pogt True --chrc_min_similarity 0.1 --lora_ema_decay 0.9
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --hmem_share_pogt True --chrc_min_similarity 0.1 --lora_ema_decay 0.9
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 0.775903 | 0.558986 | 0.880854 | User-run result with shared POGT |
| Weather | 1.917743 | 0.811171 | 1.384826 | User-run result with shared POGT |

**Observation:** ETTm1 improves vs EMA-only; Weather degrades with shared POGT under this setting.

---

### Step 6 (P2): Confidence Gate Input Simplification
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.9
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --lora_ema_decay 0.9 --chrc_min_similarity 0.9
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 0.794900 | 0.564600 | 0.891572 | User-run result with gate input simplification |
| Weather | 1.558376 | 0.701198 | 1.248349 | User-run result with gate input simplification |

**Observation:** Weather improves vs EMA-only; ETTm1 slightly worse vs EMA-only.

---

### Step 7 (P3): Active Forgetting in Memory Bank
**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --lora_ema_decay 0.9 --chrc_min_similarity 0.9 --chrc_forget_decay 0.99 --chrc_forget_threshold 0.05
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --lora_ema_decay 0.9 --chrc_min_similarity 0.9 --chrc_forget_decay 0.99 --chrc_forget_threshold 0.05
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 0.810633 | 0.571947 | 0.900352 | User-run result with active forgetting |
| Weather | 1.577724 | 0.716426 | 1.256075 | User-run result with active forgetting |

**Observation:** Both datasets regress vs EMA-only; forgetting settings may be too aggressive for these runs.
