# H-Mem 可视化实施方案
> 说明：本阶段不直接绘图，先将可视化所需数据保存为 CSV，后续由你自行绘图。

## 1. 在线学习动态曲线

### 1.1 Rolling MSE曲线
- **数据**: 每个step的MSE，滑动窗口=500
- **方法**: Frozen, Online, ER, DERpp, ACL, CLSER, MIR, SOLID, HMem
- **输出**: `rolling_mse.csv`（columns: step, method, mse, rolling_mse）
- **设计理念**: 用滚动均值平滑短期噪声，突出在线学习的长期趋势与稳定性差异。
- **目的**: 展示HMem全程稳定优势
- **解读**: HMem曲线整体更低且波动更小→在线稳定性更强；波动后回落速度体现适应性。

### 1.2 分段性能对比
- **分段**: Early (0-33%), Middle (33-66%), Late (66-100%)
- **输出**: `segment_mse.csv`（columns: segment, method, mse_mean）
- **设计理念**: 将过程分为早/中/晚三段求均值，用柱状对比阶段性变化。
- **目的**: 验证memory bank积累后优势增大
- **解读**: Late段更低说明memory积累带来持续收益；Early→Late差距越大越体现增益。

---

## 2. CHRC有效性分析

### 2.1 修正前后误差散点图 ⭐
- **X轴**: Base prediction MSE (无修正)
- **Y轴**: Corrected prediction MSE (修正后)
- **参考线**: y=x对角线
- **输出**: `chrc_scatter.csv`（columns: step, horizon, mse_base, mse_corrected）
- **设计理念**: 散点对比修正前后误差，并以对角线作为改进基线。
- **目的**: 点在对角线下方 → 修正有效
- **解读**: 点在y=x下方为改进，上方为过修正；离对角线越远代表修正幅度越大。

### 2.2 修正幅度分布
- **数据**: ||correction||₂ 的分布
- **输出**: `correction_norm.csv`（columns: step, horizon, correction_norm）
- **设计理念**: 用直方图/可选KDE展示修正幅度分布与长尾。
- **目的**: 展示大部分为小修正，少数大修正
- **解读**: 主峰靠近0说明多数为小修正；长尾表示少量大修正/突变时刻。

### 2.3 检索相似度分布
- **数据**: top-k检索的cosine similarity
- **输出**: `retrieval_similarity.csv`（columns: step, mean_similarity）
- **设计理念**: 统计检索相似度分布，衡量检索质量与稳定性。
- **目的**: 高相似度说明检索质量好
- **解读**: 分布越靠近1检索越好；偏低或分散说明检索质量不足。

---

## 3. 组件效果分析

### 3.1 Horizon Mask效果 ⭐
- **X轴**: Horizon位置 (1, 2, ..., 96)
- **Y轴**: 平均修正幅度 或 MSE改善
- **输出**: `horizon_effect.csv`（columns: horizon, metric, value）
- **设计理念**: 按预测步长绘制改进或误差曲线，观察近远期差异。
- **目的**: 验证近期horizon修正更强
- **解读**: 短期修正/改进更强符合Horizon Mask设计；曲线平坦说明效果不明显。

### 3.2 Time Buckets效果
- **数据**: 4个bucket各自的MSE分布
- **输出**: `time_buckets_mse.csv`（columns: bucket, sample_id, mse）
- **设计理念**: 按时间桶做箱线图，对比不同时间段误差分布。
- **目的**: 展示时间分桶捕捉不同时段模式
- **解读**: 不同桶分布差异明显→时段模式不同；相近则分桶收益有限。


---


### 3.3 运行效率-精度权衡
- **数据**: `runtime_summary.csv` + `test_summary.csv`（x: ms/step 或 samples/sec，y: mse/mae，size: params/memory）
- **设计理念**: 气泡图同时呈现速度、精度与资源占用，便于三维权衡。
- **解读**: x 轴越优（ms/step 越小或 samples/sec 越大），y 越低越好；气泡越大代表资源开销更高。

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
