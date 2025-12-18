# H-Mem: Horizon-Bridging Neural Memory Network for Online Time Series Forecasting

**Date:** 2025-12-12
**Type:** Method Proposal
**Based on:** Synthesis of Titans, Proceed, TAFAS, FSNet, and OneNet.

---

## 1. 核心动机 (Motivation)
当前在线时间序列预测 (Online TSF) 领域存在两个主要的“割裂” (Gap)，限制了模型在非平稳环境下的表现：

1.  **反馈延迟的割裂 (Feedback Delay Gap):** 
    *   *问题:* 预测未来 $H$ 步时，必须等待 $H$ 步之后才能获得完全的 Ground Truth (GT) 用来更新模型 (Proceed, TAFAS)。
    *   *现状:* 现有的 Fine-tuning 方法要么忽略这个延迟（导致更新滞后），要么只利用部分真实值 (POGT) 做简单的校准，缺乏深度参数更新。
2.  **记忆形式的割裂 (Memory Representation Gap):**
    *   *问题:* 如何在适应新模式的同时不遗忘旧模式？
    *   *现状:* FSNet 使用关联记忆存储梯度，但容易受噪声干扰；Titans 提出了“神经记忆”将历史压缩进权重，但在 TSF 中尚未被充分利用。

**H-Mem 核心理念:** 利用 **神经记忆 (Neural Memory)** 实时“消化”刚刚到达的部分真实值 (POGT) 以生成即时适配参数，并结合 **跨视窗检索 (Cross-Horizon Retrieval)** 从历史记忆中找回长期的纠偏模式，从而连接“当前观测”与“未来预测”的鸿沟。

---

## 2. 模型架构 (Architecture)

H-Mem 由三个互补的模块组成：

### A. 冻结的主干网络 (Frozen Backbone)
*   **作用:** 提供稳健的通用时序特征提取能力，防止灾难性遗忘。
*   **实现:** 使用预训练好的 SOTA 模型（如 PatchTST, iTransformer），在在线阶段参数完全冻结 ($F_	heta$)。

### B. 短期神经记忆适配器 (Short-term Neural Memory Adapter - SNMA)
*   **灵感来源:** Titans (Neural Memory), FSNet (Fast weights)。
*   **作用:** 利用最近观测到的部分数据 (POGT) 进行**即时**适应。
*   **机制:** 
    *   维护一个轻量级的循环神经状态 $M_t$。
    *   计算 POGT 的“惊奇度” (Surprise/Gradient)，以此更新 $M_t$。
    *   通过 Hypernetwork 将 $M_t$ 映射为 Backbone 的 LoRA (Low-Rank Adaptation) 参数 $\Delta \theta_t$。
    *   *效果:* 解决了协变量漂移 (Covariate Shift)，使模型能立即适应当前的输入分布变化。

### C. 跨视窗检索纠偏模块 (Cross-Horizon Retrieval Corrector - CHRC)
*   **灵感来源:** Proceed (Concept Drift), TAFAS (POGT utilization)。
*   **作用:** 解决**反馈延迟**问题。由于当前无法看到未来 $H$ 步的真值，我们去历史中寻找“类似情况”。
*   **机制:** 
    *   维护一个**误差记忆库 (Error Memory Bank)**，存储 `{Key: 历史POGT特征, Value: 历史全视窗残差}`。
    *   **检索 (Retrieve):** 用当前的 POGT 特征作为 Query，检索历史上最相似的 $K$ 个片段。
    *   **纠偏 (Correct):** 加权平均检索到的历史残差，作为对当前预测的补充修正项 $\hat{E}_{future}$。
    *   *效果:* 捕捉模型本身无法通过梯度下降快速学会的、周期性的或复杂的长程偏差。

---

## 3. 详细工作流程 (Workflow)

假设当前时刻为 $t$，预测视窗为 $H$，已观测部分真实值长度为 $P$ (POGT)。

1.  **基础预测:** 
    $$
    \hat{Y}_{base} = F_\theta(X_t) $$
2.  **神经记忆更新 & 参数生成 (SNMA):**
    *   计算 POGT 上的梯度: $g_p = \nabla \mathcal{L}(F_\theta(X_{t-P:t}), Y_{t-P:t})$
    *   更新记忆状态: $M_t = \text{Update}(M_{t-1}, g_p)$
    *   生成适配参数: $\Delta \theta_t = \text{HyperNet}(M_t)$
    *   生成适配预测: $\hat{Y}_{adapt} = F_{\theta + \Delta \theta_t}(X_t)$
3.  **历史检索纠偏 (CHRC):**
    *   Query = Feature($Y_{t-P:t}$)
    *   $\\hat{E}_{future} = \text{RetrieveAndAggregate}(\text{MemoryBank}, \text{Query})$
4.  **最终融合:**
    $$
    \hat{Y}_{final} = \hat{Y}_{adapt} + \lambda \cdot \hat{E}_{future} $$
    *(其中 $\\lambda$ 为动态置信度门控)*
5.  **延迟记忆入库:**
    *   当时刻 $t+H$ 到达，真实值 $Y_{t:t+H}$ 完全揭晓后，计算真实残差 $E_t$，并将 `{Feature(POGT), E_t}` 存入误差记忆库。

---

## 4. 优势对比 (Why H-Mem?)

| 维度 | FSNet / OneNet | Proceed / TAFAS | **H-Mem (Proposed)** |
| :--- | :--- | :--- | :--- |
| **适应方式** | 梯度下降 / 集成权重 | 预测漂移 / 简单校准 | **神经权重生成 + 检索增强** |
| **抗噪性** | 弱 (梯度直接更新参数) | 中 (受限于漂移预测准确度) | **强** (神经记忆平滑噪声 + 冻结主干) |
| **反馈延迟** | 忽略 (滞后更新) | 尝试解决 (预测或校准) | **彻底解决** (利用检索历史残差填补延迟空白) |
| **记忆能力** | 有限 (关联记忆/无) | 无 | **双重记忆** (参数化神经记忆 + 非参历史库) |

## 5. 关键技术细节 (Implementation Details)

*   **轻量化:** HyperNet 仅生成 Linear Layer 的 LoRA 参数 (Rank=4/8)，计算开销极低。
*   **即插即用:** 对 Backbone 不敏感，可套用在 iTransformer, PatchTST 等任何 SOTA 模型上。
*   **冷启动:** 初期记忆库为空时，退化为纯 SNMA 模式；随着时间推移，CHRC 模块逐渐生效，性能随数据量积累而提升 (Lifelong Learning)。
