# Experiment Comparison Notes

This document summarizes the available online methods in the current codebase and checks whether their experiment workflows match HMem for fair comparison.

**Date:** 2025-12-22
**Status:** Comparison Design

---

## 1. Available Online Methods

The runner selects the experiment class as `Exp_<online_method>`.

| Method | Exp Class | Source File | Description |
|--------|-----------|-------------|-------------|
| Online | `Exp_Online` | `exp/exp_online.py` | Naive online gradient descent |
| ER | `Exp_ER` | `exp/exp_online.py:337` | Experience Replay |
| DERpp | `Exp_DERpp` | `exp/exp_online.py:361` | Dark Experience Replay++ |
| FSNet | `Exp_FSNet` | `exp/exp_online.py:383` | Learning Fast and Slow (ICLR 2023) |
| OneNet | `Exp_OneNet` | `exp/exp_online.py:417` | Online Ensembling (NeurIPS 2024) |
| SOLID | `Exp_SOLID` | `exp/exp_solid.py` | Sample-level Contextualized Adapter (KDD 2024) |
| Proceed | `Exp_Proceed` | `exp/exp_proceed.py` | Proactive Model Adaptation (KDD 2025) |
| ACL | `Exp_ACL` | `exp/exp_online.py` | Adaptive Continual Learning (ACL) |
| CLSER | `Exp_CLSER` | `exp/exp_online.py` | Complementary Learning System - Experience Replay |
| MIR | `Exp_MIR` | `exp/exp_online.py` | Maximally Interfered Retrieval |
| HMem | `Exp_HMem` | `exp/exp_hmem.py` | Horizon-Bridging Neural Memory (Ours) |

---

## 2. Workflow Alignment Summary

### 2.1 Core Online Loop Comparison

```
Exp_Online 标准流程:
┌─────────────────────────────────────────────────────────┐
│ for (recent_batch, current_batch) in online_loader:     │
│   1. model.train()                                      │
│   2. _update_online(recent_batch)  ← 用历史数据更新     │
│   3. model.eval()                                       │
│   4. forward(current_batch)        ← 用当前数据评估     │
└─────────────────────────────────────────────────────────┘
```

| Method | 继承 Exp_Online | 使用 (recent, current) | update_valid() | 信息泄露风险 |
|--------|-----------------|------------------------|----------------|--------------|
| Online | Yes | Yes | Yes | No |
| ER | Yes | Yes | Yes | No |
| DERpp | Yes | Yes | Yes | No |
| FSNet | Yes | Yes | Yes | No |
| OneNet | Yes | Yes | Yes | No |
| SOLID | Yes (但重写 online) | **No** (自定义检索) | **No** | No |
| Proceed | Yes | Yes | Yes | No |
| ACL | Yes | Yes | Yes | No |
| CLSER | Yes | Yes | Yes | No |
| MIR | Yes | Yes | Yes | No |
| **HMem** | Yes | Yes | Yes | No |

### 2.2 Methods that share the Exp_Online loop

These methods inherit `Exp_Online` and use the same high-level online loop:
- Data loader uses `Dataset_Recent` and yields `(recent_batch, current_batch)`.
- Per step: update on `recent_batch`, then evaluate on `current_batch`.

Methods in this group:
- **Online**: Naive gradient descent on backbone
- **ER**: + Replay buffer (size=500, sample=8), loss += 0.2 * replay_loss
- **DERpp**: + Replay buffer + distillation loss on stored predictions
- **FSNet**: + Fast/slow learning mechanism with gradient storage
- **OneNet**: + Ensemble of fast/slow models with learned weights
- **ACL**: + Replay buffers + feature consistency + hint distillation
- **CLSER**: + Dual EMA teacher models + consistency regularization
- **MIR**: + Replay samples selected by maximal interference

### 2.3 Proceed (partial alignment)

Proceed inherits `Exp_Online` but changes update behavior:
- During `online()`, it freezes the adapter and only updates selected parts.
- During `update_valid()`, it alternates which parts are frozen between recent/current updates.
- Uses PEFT (Parameter-Efficient Fine-Tuning) adapter architecture.

This changes which parameters are updated compared to HMem.

### 2.4 SOLID (different workflow)

**SOLID does NOT use the standard `Exp_Online` recent/current loop.** Instead:
```python
# SOLID 流程 (exp_solid.py:123-216)
for i, batch in enumerate(test_loader):
    # 1. 检索历史相似样本
    distance_pairs = F.pairwise_distance(batch_x, lookback_x)
    selected_indices = distance_pairs.topk(selected_data_num, largest=False)

    # 2. 用相似样本更新 head
    loss = criterion(self.forward(selected_x), selected_y)
    loss.backward()

    # 3. 评估当前样本
    outputs = self.forward(batch)

    # 4. 可选：恢复预训练权重 (非 continual 模式)
    if not self.args.continual:
        self.final_head.load_state_dict(pretrained_state_dict)
```

Key differences:
- Uses test data directly (not recent/current pairs)
- Sample selection by L2 distance similarity
- Per-step head restoration (transductive learning)

### 2.5 HMem (distinct workflow)

HMem extends `Exp_Online` but introduces its own online logic:
- **Warmup phase**: `update_valid()` populates CHRC memory bank with validation data
- **Delayed memory updates**: Wait for ground truth horizon before storing errors
- **POGT extraction**: Uses `batch_x[-pogt_len:]` as causal observation (no leakage)
- **Two-phase training**: Optional SNMA warmup before joint SNMA+CHRC training

```python
# HMem 流程 (exp_hmem.py)
# Phase 1: Validation warmup
update_valid()  # Populate memory bank

# Phase 2: Online learning
for (recent_batch, current_batch) in online_loader:
    # Extract POGT from recent observations (causal)
    pogt = batch_x[:, -pogt_len:, :]

    # Update with recent batch
    _update_online(recent_batch)

    # Delayed memory bank update (after pred_len steps)
    _process_delayed_updates()

    # Predict on current batch
    prediction = model(current_batch, pogt=pogt)
```

---

## 3. Data Split and Preprocessing

### 3.1 Border Type

All methods should use `--border_type online` for fair comparison:

```python
# settings.py:get_borders()
if border_type == 'online':
    # ETTm1: 20% train, 5% val, 75% test
    border1s = [0, 4*30*24*4 - seq_len, 5*30*24*4 - seq_len]
    border2s = [4*30*24*4, 5*30*24*4, 20*30*24*4]
```

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 20% | Pretraining backbone |
| Val | 5% | Online warmup / adaptation |
| Test | 75% | Online evaluation |

### 3.2 Pretrained Checkpoint

All methods load the same pretrained backbone:
```bash
--pretrain --only_test
```

Checkpoint path: `checkpoints/{dataset}_{seq_len}_{pred_len}_{model}/`

---

## 4. Fairness Considerations

### 4.1 Controlled Variables (Must Be Same)

| Variable | Recommended Value | Notes |
|----------|-------------------|-------|
| `--border_type` | `online` | 20/5/75 split |
| `--model` | `iTransformer` | Same backbone |
| `--seq_len` | `512` | Same lookback |
| `--pred_len` | `96` | Same horizon |
| `--pretrain` | `True` | Use pretrained model |
| `--only_test` | `True` | No further pretraining |
| Checkpoint | Same path | Same initialization |

### 4.2 Method-Specific Variables (Can Differ)

| Variable | Notes |
|----------|-------|
| `--online_learning_rate` | Each method has its optimal LR |
| Buffer/Memory size | ER/DERpp: 500, HMem: configurable |
| Extra modules | Adapter, SNMA, CHRC, etc. |

### 4.3 Key Fairness Issues

#### Issue 1: Validation Set Usage

| Method | Uses update_valid() | Memory/Adaptation on Val |
|--------|---------------------|--------------------------|
| Online | Yes | Backbone update |
| ER/DERpp | Yes | Backbone + Buffer fill |
| SOLID | **No** | None |
| Proceed | Yes | Adapter warmup |
| HMem | Yes | Memory bank population |

**Recommendation**:
- Option A: All methods skip `update_valid()` (set `--skip_valid_update`)
- Option B: All methods use `update_valid()` (modify SOLID to add warmup)
- Option C: Report SOLID separately as different protocol

#### Issue 2: Learning Rate Sensitivity

Different methods have different optimal learning rates:

| Method | Typical LR Range | Notes |
|--------|------------------|-------|
| Online | 1e-5 ~ 3e-5 | Higher LR can diverge |
| ER/DERpp | 1e-5 ~ 3e-5 | Similar to Online |
| SOLID | 1e-4 ~ 1e-3 | Head-only update allows higher LR |
| Proceed | 1e-5 ~ 3e-5 | Adapter LR |
| HMem | 1e-5 ~ 3e-5 | CHRC encoder LR |

**Recommendation**: Use each method's tuned optimal LR (report in paper).

#### Issue 3: Memory/Buffer Capacity

| Method | Memory Type | Default Size |
|--------|-------------|--------------|
| ER | Replay buffer | 500 samples |
| DERpp | Replay buffer | 500 samples |
| ACL | Replay buffers | 500 + 50 samples |
| CLSER | Replay buffer | 500 samples |
| MIR | Replay buffer | 500 samples |
| HMem | Error memory bank | 1000 entries (4 buckets) |

**Recommendation**: Report memory footprint alongside performance.

---

## 5. Recommended Experiment Commands

### 5.1 ETTm1 Dataset

```bash
# Frozen Baseline (no online update)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain

# Online (naive gradient descent)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 3e-5

# ER (Experience Replay)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 3e-5

# DERpp (Dark Experience Replay++)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method DERpp --only_test --pretrain --online_learning_rate 3e-5

# ACL
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ACL --only_test --pretrain --online_learning_rate 3e-5

# CLSER
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method CLSER --only_test --pretrain --online_learning_rate 3e-5

# MIR
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method MIR --only_test --pretrain --online_learning_rate 3e-5

# SOLID (Sample-level Contextualized Adapter)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method SOLID --only_test --pretrain --online_learning_rate 1e-4

# Proceed (Proactive Model Adaptation, KDD 2025)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Proceed --only_test --pretrain --online_learning_rate 3e-5 --freeze --concept_dim 200 --bottleneck_dim 32 --tune_mode down_up

# HMem (Ours, P2+P7+P8 config)
python -u run.py --dataset ETTm1 --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets True --chrc_bucket_num 4
```

### 5.2 Weather Dataset

```bash
# Frozen Baseline
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --only_test --pretrain

# Online
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Online --only_test --pretrain --online_learning_rate 3e-5

# ER
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ER --only_test --pretrain --online_learning_rate 3e-5

# DERpp
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method DERpp --only_test --pretrain --online_learning_rate 3e-5

# ACL
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method ACL --only_test --pretrain --online_learning_rate 3e-5

# CLSER
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method CLSER --only_test --pretrain --online_learning_rate 3e-5

# MIR
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method MIR --only_test --pretrain --online_learning_rate 3e-5

# SOLID
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method SOLID --only_test --pretrain --online_learning_rate 1e-4

# Proceed
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method Proceed --only_test --pretrain --online_learning_rate 3e-5 --freeze --concept_dim 200 --bottleneck_dim 32 --tune_mode down_up

# HMem (P2+P7, no buckets for Weather)
python -u run.py --dataset Weather --border_type online --model iTransformer --seq_len 512 --pred_len 96 --itr 1 --online_method HMem --only_test --pretrain --online_learning_rate 1e-5 --use_snma False --use_chrc True --retrieval_top_k 5 --chrc_aggregation softmax --chrc_trust_threshold 0.5 --chrc_gate_steepness 10.0 --chrc_use_horizon_mask True --chrc_horizon_mask_mode exp --chrc_horizon_mask_decay 0.98 --chrc_horizon_mask_min 0.2 --chrc_use_buckets False
```

---

## 6. Expected Results Template

| Method | ETTm1 MSE | ETTm1 MAE | Weather MSE | Weather MAE | Notes |
|--------|-----------|-----------|-------------|-------------|-------|
| Frozen | - | - | - | - | No online update |
| Online | - | - | - | - | Naive GD |
| ER | - | - | - | - | + Replay buffer |
| DERpp | - | - | - | - | + Distillation |
| ACL | - | - | - | - | Replay + consistency + hint |
| CLSER | - | - | - | - | Dual EMA + consistency |
| MIR | - | - | - | - | Max interference replay |
| SOLID | - | - | - | - | Different protocol |
| Proceed | - | - | - | - | PEFT adapter (KDD 2025) |
| **HMem** | **0.703** | - | **1.424** | - | P2+P7+P8 |

---

## 7. Open Questions

1. **SOLID Protocol**: Should SOLID be reported in a separate table due to its fundamentally different workflow (sample retrieval vs. recent/current)?

2. **Proceed Inclusion**: Proceed is another contribution of this project. Should it be included as a baseline or excluded to avoid self-comparison?

3. **Multiple Runs**: Should we report mean±std over multiple seeds (e.g., `--itr 3`)?

4. **Additional Datasets**: Which other datasets to include? (ETTh1, ETTh2, ECL, Traffic)

5. **Ablation Study**: Should we include HMem ablations (CHRC-only, SNMA-only, different P combinations)?

