# H-Mem Improvement Experiment Results

**Date:** 2025-12-19  
**Scope:** Post-change validation for recent architectural fixes  
**Purpose:** Keep new experiments separate from the historical baseline report

---

## P0-1: POGT Causality Fix (Prediction uses observed `recent_batch` only)

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

## P0-2: SNMA Memory Detach (Preserve memory across steps)

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

## Pending (Not Yet Evaluated)

## P1: CHRC Absolute Similarity Abstention

**Commands:** (Please confirm exact `--chrc_min_similarity` value used.)
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True --chrc_min_similarity 0.1
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma False --use_chrc True --chrc_aggregation weighted_mean --chrc_min_similarity <value>
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 0.826357 | 0.577490 | 0.909042 | User-run result with CHRC abstention |
| Weather | 1.560324 | 0.709551 | 1.249129 | User-run result with CHRC abstention |

**Observation:** Weather improves vs P0 causal baseline; ETTm1 returns close to static baseline.

---

## Pending (Not Yet Evaluated)

- P1: HyperNetwork volatility reduction (EMA / target module reduction)

---

## Baseline Re-Run (Online Method)

**Commands:**
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 1e-4
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 1e-4
```

| Dataset | MSE | MAE | RMSE | Notes |
| :--- | :--- | :--- | :--- | :--- |
| ETTm1 | 1.193117 | 0.697323 | 1.092299 | User rerun |
| Weather | 2.983787 | 1.040027 | 1.727364 | User rerun |
