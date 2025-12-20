# H-Mem 2.0 (CHRC-Centric) 实验记录

**日期:** 2025-12-20  
**关联规划:** `@1220Hmem_improve.md`  
**阶段:** P0 (Baseline + 诊断)

---

## 1. 变更概览（P0）

目标：对齐 `@1220Hmem_improve.md` 的 P0 任务，确保 CHRC-only 基线与 `chrc_min_similarity` 诊断可用，并强制因果 POGT。

**代码改动摘要：**
- CHRC 检索相似度日志（排查 `chrc_min_similarity` 无效原因）
  - `util/error_bank.py` 添加打印：min/max/mean/valid_ratio
- SNMA 默认关闭（CHRC-only 为默认）
  - `run.py` / `settings.py` / `adapter/hmem.py` / `exp/exp_hmem.py`
- POGT 来源可配置（默认 `batch_x`，因果）
  - 新增参数 `--hmem_pogt_source`，支持 `batch_x`/`batch_y`
  - `exp/exp_hmem.py` 预测与更新统一使用该来源

---

## 2. 实验清单（P0）

### Exp-A: CHRC-only 基线（因果 POGT）

**ETTm1**
```
python -u run.py --dataset ETTm1 --border_type online --model iTransformer \
  --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 1e-5 \
  --use_snma False --use_chrc True --hmem_pogt_source batch_x
```

**Weather**
```
python -u run.py --dataset Weather --border_type online --model iTransformer \
  --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 5e-6 \
  --use_snma False --use_chrc True --chrc_aggregation weighted_mean \
  --hmem_pogt_source batch_x
```

**结果记录：**
- ETTm1: MSE=0.777628  MAE=0.561764  RMSE=0.881832
- Weather: MSE=1.770241  MAE=0.770953  RMSE=1.330504

---

### Exp-B: `chrc_min_similarity` 诊断（因果 POGT）

**ETTm1**
```
python -u run.py --dataset ETTm1 --border_type online --model iTransformer \
  --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 1e-5 \
  --use_snma False --use_chrc True --hmem_pogt_source batch_x \
  --chrc_min_similarity 0.6
```

**Weather**
```
python -u run.py --dataset Weather --border_type online --model iTransformer \
  --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 5e-6 \
  --use_snma False --use_chrc True --chrc_aggregation weighted_mean \
  --hmem_pogt_source batch_x --chrc_min_similarity 0.6
```

**结果记录：**
- ETTm1: MSE=0.778484  MAE=0.561994  RMSE=0.882317
- Weather: MSE=1.829211  MAE=0.785710  RMSE=1.352483

**相似度日志观察（从 console 记录）：**
- ETTm1: min=0.969-0.973  max=0.995  mean=0.982-0.984  valid_ratio=1.000
- Weather: min=0.969-0.973  max=0.995  mean=0.982-0.984  valid_ratio=1.000

---

### Exp-C: POGT 泄漏对照（batch_y）

将 `--hmem_pogt_source batch_x` 改为 `batch_y`，作为对照。

**ETTm1**
```
python -u run.py --dataset ETTm1 --border_type online --model iTransformer \
  --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 1e-5 \
  --use_snma False --use_chrc True --hmem_pogt_source batch_y
```

**Weather**
```
python -u run.py --dataset Weather --border_type online --model iTransformer \
  --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 5e-6 \
  --use_snma False --use_chrc True --chrc_aggregation weighted_mean \
  --hmem_pogt_source batch_y
```

**结果记录：**
- ETTm1: MSE=____  MAE=____  RMSE=____
- Weather: MSE=____  MAE=____  RMSE=____

---

## 3. 观察与结论（待填写）

- Baseline vs `chrc_min_similarity` 变化：____
- 相似度分布是否过窄：____
- `batch_x` vs `batch_y` 差异（泄漏收益）：____

---

## P1 Results (Dual-Key Comparison)

ETTm1
- chrc_use_dual_key=False: MSE=0.777628  MAE=0.561764  RMSE=0.881832
- chrc_use_dual_key=True: MSE=0.777710  MAE=0.560736  RMSE=0.881879

Weather
- chrc_use_dual_key=False: MSE=1.770241  MAE=0.770953  RMSE=1.330504
- chrc_use_dual_key=True: MSE=1.916031  MAE=0.815432  RMSE=1.384208
