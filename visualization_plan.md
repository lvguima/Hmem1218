# H-Mem 可视化实施方案
> 说明：本阶段不直接绘图，先将可视化所需数据保存为 CSV，后续由你自行绘图。

## 1. 在线学习动态曲线

### 1.1 Rolling MSE曲线
- **数据**: 每个step的MSE，滑动窗口=500
- **方法**: Frozen, Online, ER, DERpp, ACL, CLSER, MIR, SOLID, HMem
- **输出**: `rolling_mse.csv`（columns: step, method, mse, rolling_mse）
- **目的**: 展示HMem全程稳定优势

### 1.2 分段性能对比
- **分段**: Early (0-33%), Middle (33-66%), Late (66-100%)
- **输出**: `segment_mse.csv`（columns: segment, method, mse_mean）
- **目的**: 验证memory bank积累后优势增大

---

## 2. CHRC有效性分析

### 2.1 修正前后误差散点图 ⭐
- **X轴**: Base prediction MSE (无修正)
- **Y轴**: Corrected prediction MSE (修正后)
- **参考线**: y=x对角线
- **输出**: `chrc_scatter.csv`（columns: step, horizon, mse_base, mse_corrected）
- **目的**: 点在对角线下方 → 修正有效

### 2.2 修正幅度分布
- **数据**: ||correction||₂ 的分布
- **输出**: `correction_norm.csv`（columns: step, horizon, correction_norm）
- **目的**: 展示大部分为小修正，少数大修正

### 2.3 检索相似度分布
- **数据**: top-k检索的cosine similarity
- **输出**: `retrieval_similarity.csv`（columns: step, mean_similarity）
- **目的**: 高相似度说明检索质量好

---

## 3. 组件效果分析

### 3.1 Horizon Mask效果 ⭐
- **X轴**: Horizon位置 (1, 2, ..., 96)
- **Y轴**: 平均修正幅度 或 MSE改善
- **输出**: `horizon_effect.csv`（columns: horizon, metric, value）
- **目的**: 验证近期horizon修正更强

### 3.2 Time Buckets效果
- **数据**: 4个bucket各自的MSE分布
- **输出**: `time_buckets_mse.csv`（columns: bucket, sample_id, mse）
- **目的**: 展示时间分桶捕捉不同时段模式


---

## 4. 实现优先级

| 优先级 | 可视化 | 说明 |
|--------|--------|------|
| P0 | 修正前后误差散点图 | 最直观证明CHRC有效 |
| P0 | Horizon Mask效果 | 展示核心组件作用 |
| P1 | Rolling MSE曲线 | 展示在线学习动态 |
| P1 | 分段性能对比 | 展示memory积累效果 |
| P2 | 修正幅度分布 | 补充分析 |
| P2 | Time Buckets效果 | 补充分析 |

---
