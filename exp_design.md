# H-Mem Diagnosis Experiment Plan

**Date:** 2025-12-14
**Objective:** Diagnose the root cause of H-Mem's underperformance (MSE 0.998) compared to ER baseline (MSE 0.757) on ETTm1.
**Focus:** Memory capacity, learning rate sensitivity, component ablation (SNMA/CHRC), and normalization issues.

---

## 1. Baseline Verification (Ablation Studies)

### Exp 1: Pure SNMA (No CHRC)
**Goal:** Isolate the Short-term Neural Memory Adapter. Determine if the "HyperNetwork + LoRA" mechanism itself is learning effectively or creating noise.
**Expectation:** If MSE >> 0.757, SNMA hyperparameters (LR, Rank, Momentum) are likely the bottleneck.
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.0001 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc False
```

### Exp 2: Pure CHRC (No SNMA) - "Retrieval Only"
**Goal:** Isolate the Retrieval Corrector. See if retrieving historical errors *alone* helps or hurts.
**Expectation:** If result is poor, Memory Bank capacity or feature matching (non-stationarity) is the issue.
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.0001 --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 --freeze True --use_snma False --use_chrc True --chrc_ramp_steps 500 --chrc_min_entries 50 --chrc_max_correction 5.0
```

---

## 2. Capacity & Stability Tuning

### Exp 3: High Capacity + Low LR (Recommended Fix)
**Goal:** Address the two most likely culprits: insufficient history (Capacity=100 -> 2000) and aggressive updates (LR 1e-4 -> 1e-5).
**Hypothesis:** ETTm1 requires longer context to find useful error patterns, and LoRA needs subtle updates to avoid disrupting the pre-trained backbone.
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --lora_rank 16 --lora_alpha 32.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True --chrc_ramp_steps 500 --chrc_min_entries 50 --chrc_max_correction 5.0
```

### Exp 4: Aggressive Retrieval (Larger Top-K & Capacity)
**Goal:** Test if CHRC needs *more* evidence to work. Increasing `top_k` to 10 and Capacity to 4000 (approx 1.5 months of data).
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00001 --lora_rank 8 --memory_capacity 4000 --retrieval_top_k 10 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True --chrc_ramp_steps 500 --chrc_min_entries 100 --chrc_max_correction 5.0
```

---

## 3. Comparison Baseline

### Exp 5: ER Baseline (Reproduce for Fairness)
**Goal:** Run the ER baseline in the *exact same environment* to confirm the target score (0.757). Note the extremely low LR.
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 0.0000003 --pin_gpu True
```

---

## 4. Feature Engineering Diagnosis

### Exp 6: High Rank LoRA
**Goal:** Maybe Rank=8 is too small to capture the necessary adaptation? Try Rank=32.
```bash
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 0.00005 --lora_rank 32 --lora_alpha 64.0 --memory_dim 512 --bottleneck_dim 64 --memory_capacity 2000 --hmem_warmup_steps 100 --freeze True --use_snma True --use_chrc True --chrc_ramp_steps 500 --chrc_max_correction 5.0
```

---

## Analysis Strategy

1.  **If Exp 1 (Pure SNMA) fails:** The issue is in `adapter/module/neural_memory.py` or `lora.py`. We need to check if gradients are actually flowing or if the HyperNetwork is initializing to zero/garbage.
2.  **If Exp 1 works but Exp 3 (Full) fails:** The issue is definitively in `CHRC`. The retrieval is fetching "poisonous" corrections. We might need to implement **Instance Normalization on POGT features** before storage/retrieval.
3.  **If Exp 3 works:** It was just a parameter tuning issue (Capacity/LR).


 1. SNMA-only（关 CHRC，降 lr）
     python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain
     --online_learning_rate 3e-6 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 500 --retrieval_top_k 5 --hmem_warmup_steps 200
     --freeze True --use_snma True --use_chrc False
  2. CHRC 线性聚合 + 较低 lr
     python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain
     --online_learning_rate 1e-5 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 500 --retrieval_top_k 5 --hmem_warmup_steps 200
     --freeze True --use_snma True --use_chrc True --chrc_aggregation weighted_mean
  3. CHRC softmax 但缓和温度 + 大容量
     python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain
     --online_learning_rate 1e-5 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 1000 --retrieval_top_k 10 --hmem_warmup_steps
     300 --freeze True --use_snma True --use_chrc True --chrc_aggregation softmax --chrc_temperature 0.5
  4. CHRC 渐进启用（需你已加 ramp 逻辑）
     python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain
     --online_learning_rate 1e-5 --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 --memory_capacity 500 --retrieval_top_k 5 --hmem_warmup_steps 200
     --freeze True --use_snma True --use_chrc True --chrc_ramp_steps 1000 --chrc_min_entries 20
  5. 对照基线 ER（复现你更好的结果）
     python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain
     --online_learning_rate 3e-7 --pin_gpu True --save_opt