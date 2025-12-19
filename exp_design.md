# H-Mem Comprehensive Experimental Design & Diagnosis Plan

**Date:** 2025-12-18
**Status:** Phase 2 (Deep Dive & Robustness Testing)
**Author:** Gemini Agent (on behalf of User)

---

## 1. Executive Summary & Motivation

**Current Status:**
1.  **ETTm1 (Periodic/Stable):** H-Mem achieves SOTA (MSE 0.8265 vs Baseline 0.8293). Key factors identified: **Low LR** ($10^{-5}$) and **Large Memory Capacity** (4000).
2.  **Weather (Chaotic/Noisy):** H-Mem suffers from "Catastrophic Noise Amplification" (MSE 3.48 vs Baseline 1.71). SNMA appears to overfit high-frequency noise.

**Objective:**
To systematically dissect H-Mem's components (SNMA vs. CHRC), quantify the "Stability vs. Plasticity" trade-off, and establish a robust configuration that works across contradictory data regimes (Periodic vs. Chaotic).

---

## 2. Experiment Series A: Component Isolation (The Anatomy Study)

**Hypothesis:** We need to strictly quantify the marginal gain (or loss) of each module independently before combining them.
**Datasets:** `ETTm1` (Representing Stability), `Weather` (Representing Chaos).

### A1. Baseline (Static Backbone)
*Control group. Pure iTransformer without any online adaptation.*
```bash
# ETTm1
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method None --only_test --pretrain 
MSE:0.829312, MAE:0.583861, RMSE:0.910666


# Weather
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method None --only_test --pretrain
MSE:1.714112, MAE:0.734446, RMSE:1.309241
```
**Note:** `online_method None` runs the standard offline `Exp_Main.test()` (no online update loop). Use the online baselines below to verify the update→predict pipeline.

### A2. Pure SNMA (The "Adapter" Test)
*Testing Short-term Adaptation. Can a HyperNetwork learn to shift weights based on POGT?*
*   **Key Config:** `use_snma=True`, `use_chrc=False`, `freeze=True`.
*   **Critical Variant:** Testing High vs. Low LR to find the "Safe Zone".
```bash
# ETTm1 - Conservative SNMA
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc False --freeze True --hmem_warmup_steps 200 
MSE: 0.838189 | MAE: 0.584677 | RMSE: 0.915527

# Weather - Conservative SNMA (Crucial check for noise overfitting)
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --use_snma True --use_chrc False --freeze True --hmem_warmup_steps 500 --lora_dropout 0.1
MSE: 2.075640 | MAE: 0.810040 | RMSE: 1.440708
```

### A3. Pure CHRC (The "Memory" Test)
*Testing Retrieval. Can historical errors correct current predictions without weight updates?*
*   **Key Config:** `use_snma=False`, `use_chrc=True`.
*   **Focus:** Retrieval Accuracy.
```bash
# ETTm1 - Large Capacity
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --freeze True --memory_capacity 4000 --retrieval_top_k 10 --chrc_aggregation softmax
 MSE: 0.835609 | MAE: 0.590440 | RMSE: 0.914116

# Weather - Weighted Mean (More robust to outliers)
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --freeze True --memory_capacity 2000 --retrieval_top_k 10 --chrc_aggregation weighted_mean
MSE: 1.866150 | MAE: 0.790729 | RMSE: 1.36607
```

### A4. Standard Online Baselines (Non-HMem)
*Sanity check that the online pipeline itself is correct and competitive.*
```bash
# ETTm1 - Online (no rehearsal)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 1e-4 --freeze False
MSE:1.166494, MAE:0.687838, RMSE:1.080044

# Weather - Online (no rehearsal)
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 1e-4 --freeze False
MSE:2.884080, MAE:1.011597, RMSE:1.698258

# ETTm1 - ER (rehearsal)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 1e-4 --freeze False
MSE:0.955979, MAE:0.621468, RMSE:0.977742

# Weather - ER (rehearsal)
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 1e-4 --freeze False
MSE:1.766211, MAE:0.758254, RMSE:1.328989


# ETTm1 - DERpp
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method DERpp --only_test --pretrain --online_learning_rate 1e-4 --freeze False
MSE:0.965773, MAE:0.625035, RMSE:0.982738

# Weather - DERpp
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method DERpp --only_test --pretrain --online_learning_rate 1e-4 --freeze False

# ETTm1 - SOLID (stronger online baseline, uses internal selection)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method SOLID --only_test --pretrain --freeze False
MSE:0.829228, MAE:0.583832, RMSE:0.910620

# Weather - SOLID
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method SOLID --only_test --pretrain --freeze False
MSE:1.723247, MAE:0.737336, RMSE:1.312725
```

---

## 3. Experiment Series B: Stability vs. Plasticity (Parameter Sensitivity)

**Hypothesis:** ETTm1 needs Plasticity (fast adaptation), Weather needs Stability (noise filtering). The Learning Rate and LoRA Rank are the control knobs.

### B1. The Learning Rate Sweep (ETTm1)
*Finding the "Sweet Spot" for SNMA.*
```bash
# Aggressive (1e-4) - Expected: High Variance / Fail
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-4 --use_snma True --use_chrc True

# Balanced (1e-5) - Expected: Good
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma True --use_chrc True

# Conservative (1e-6) - Expected: Too slow, close to baseline
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-6 --use_snma True --use_chrc True
```

### B2. LoRA Rank & Expressiveness (ETTm1)
*Does a larger Rank allow capturing more complex drifts, or just overfit?*
```bash
# Low Rank (4) - Regularization
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --lora_rank 4 --lora_alpha 8.0 --use_snma True --use_chrc True

# High Rank (32) - High Expressiveness
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --lora_rank 32 --lora_alpha 64.0 --use_snma True --use_chrc True
```

---

## 4. Experiment Series C: Memory Dynamics (The "Brain" Upgrade)

**Hypothesis:** The quality of CHRC depends on **Memory Density** (Capacity) and **Retrieval Wisdom** (Aggregation).

### C1. Capacity Scaling (ETTm1)
*Is "More History" always better?*
```bash
# Short Memory (500)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --memory_capacity 500 --use_snma True --use_chrc True

# Long Memory (4000) - Already proven good, validating repeatability
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --memory_capacity 4000 --use_snma True --use_chrc True
```

### C2. Retrieval Mechanism (Weather - Critical)
*Softmax is sharp (pick the best). Weighted Mean is smooth (average the neighbors). Smoothness helps in noise.*
```bash
# Softmax + Low Temp (Sharp selection)
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --chrc_aggregation softmax --chrc_temperature 0.1 --use_snma False --use_chrc True

# Weighted Mean (Smooth aggregation) - Expected winner for Weather
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 5e-6 --chrc_aggregation weighted_mean --use_snma False --use_chrc True
```

---

## 5. Experiment Series D: The "Anti-Noise" Configuration (Weather Rescue)

**Hypothesis:** To make H-Mem work on Weather, we must throttle SNMA and smooth CHRC. This is the **"Safe Mode"** configuration.

**Configuration Profile:**
*   `online_learning_rate`: **1e-6** (Extremely slow updates)
*   `hmem_warmup_steps`: **500** (Long observation before acting)
*   `lora_dropout`: **0.2** (High regularization)
*   `chrc_aggregation`: **weighted_mean** (Avoid overfitting to single historical error)
*   `pogt_ratio`: **0.25** (Only look at very recent trend, ignore stale noise)

```bash
# The "Safe Mode" Run
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain \
  --online_learning_rate 1e-6 \
  --hmem_warmup_steps 500 \
  --lora_rank 4 \
  --lora_dropout 0.2 \
  --use_snma True \
  --use_chrc True \
  --chrc_aggregation weighted_mean \
  --pogt_ratio 0.25
```

---

## 6. Analysis Framework (How to interpret results)

| Observation | Likely Cause | Action |
| :--- | :--- | :--- |
| **SNMA works on ETT, fails on Weather** | High-frequency noise in Weather POGT is tricking the HyperNet. | Increase `lora_dropout`, reduce `pogt_ratio` (denoise), or use `weighted_mean` CHRC only. |
| **CHRC degrades performance** | "Negative Transfer" - retrieving irrelevant historical errors. | Increase `retrieval_top_k` (smooth out bad matches), switch to `weighted_mean`. |
| **High LR degrades performance** | Catastrophic Forgetting / Parameter Oscillation. | Lower LR is non-negotiable for Online TSF. Stick to $10^{-5}$ or $10^{-6}$. |
| **Full H-Mem < Pure CHRC** | SNMA is destructively interfering with Backbone. | Disable SNMA, use H-Mem as a pure "Retrieval Augmented" system (RAG-only). |

---
