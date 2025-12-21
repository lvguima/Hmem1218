# H-Mem 2.1: SNMA 重访与检索机制升级

**日期:** 2025-12-21
**状态:** 探索性研究计划
**前置依赖:** 1220 CHRC 优化已完成，最佳配置为 P2+P7+P8

---

## 1. 背景与动机

### 1.1 当前状态

1220 实验取得了显著成果：
- ETTm1: 0.778 → **0.703** (-9.6%)
- Weather: 1.770 → **1.452** (-18.0%)

**核心公式**:
```
Ŷ_corrected = Ŷ_base + α(sim) × w(horizon) × Bucket[t].retrieve(POGT)
```

### 1.2 尚未解决的问题

1. **SNMA 被完全弃用**: 之前 SNMA + CHRC 组合效果差（0.814 vs 单独 0.776/0.780），但根本原因未被解决
2. **检索机制固定**: 仅使用余弦相似度，可能不是最优
3. **Weather 目标未达成**: 1.452 vs 目标 1.400，还有 3.7% 的差距

### 1.3 研究假设

**假设 1**: SNMA 失败是因为与 CHRC "争夺"同一个预测空间，导致冲突。如果让 SNMA 负责**不同层面**的适应，可能产生互补效应。

**假设 2**: 余弦相似度是**语义无关**的，仅衡量向量方向相似性。学习一个**误差预测相关**的相似度度量，可能提升检索质量。

---

## 2. 方向 A: 简化版 SNMA 重访

### 2.1 原始 SNMA 失败原因分析

原始 SNMA 架构：
```
POGT → SurpriseCalculator → Memory State → HyperNetwork → LoRA Params → Backbone Adaptation
```

**问题诊断**:

| 问题 | 原因 | 证据 |
|------|------|------|
| **与 CHRC 冲突** | SNMA 修改 backbone 预测，CHRC 在此基础上再修正 → 双重修正放大误差 | SNMA+CHRC (0.814) > 单独 (0.776/0.780) |
| **超网络过于复杂** | HyperNetwork 生成数千参数，难以稳定训练 | LR 敏感，1e-5→3e-5 导致 +12% MSE |
| **惊奇度计算不可靠** | 自监督预测任务与误差预测任务不对齐 | 需要验证 |

### 2.2 简化设计方案

**核心思想**: 不再让 SNMA 修改 backbone，而是让它学习一个**残差调整项**，与 CHRC 互补。

#### 方案 A1: SNMA 作为 CHRC 的"动态增强器"

```
                    ┌─────────────────────────┐
                    │   SNMA-Light            │
                    │   (无 HyperNetwork)      │
POGT ──► Encoder ──►│                         │──► Δ_snma (残差)
                    │   Memory State + MLP    │
                    └─────────────────────────┘
                               │
                               ▼
Ŷ_corrected = Ŷ_base + CHRC_correction + β × Δ_snma
                                           ↑
                                    小系数，避免冲突
```

**关键简化**:
1. **移除 HyperNetwork**: 不再生成 LoRA 参数
2. **直接预测残差**: Memory State → MLP → 残差 Δ
3. **小权重组合**: β 初始化为 0.1-0.2，防止与 CHRC 冲突
4. **独立学习目标**: SNMA 学习 CHRC 未能捕获的误差成分

**实现草图**:
```python
class SNMALight(nn.Module):
    """简化版 SNMA，直接预测残差"""

    def __init__(self, input_features, memory_dim=128, horizon=96, num_features=7):
        super().__init__()
        self.memory_dim = memory_dim

        # 简化的编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_features, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )

        # 记忆状态 (可学习)
        self.memory = nn.Parameter(torch.zeros(memory_dim))
        self.memory_gate = nn.Linear(memory_dim * 2, memory_dim)

        # 残差预测器
        self.residual_predictor = nn.Sequential(
            nn.Linear(memory_dim, memory_dim * 2),
            nn.GELU(),
            nn.Linear(memory_dim * 2, horizon * num_features),
        )

    def forward(self, pogt):
        # 编码 POGT
        h = self.encoder(pogt.mean(dim=1))  # [B, memory_dim]

        # 更新记忆
        gate = torch.sigmoid(self.memory_gate(
            torch.cat([h, self.memory.expand(h.size(0), -1)], dim=-1)
        ))
        self.memory.data = gate.mean(dim=0) * h.mean(dim=0) + (1 - gate.mean(dim=0)) * self.memory.data

        # 预测残差
        residual = self.residual_predictor(h)
        return residual.reshape(h.size(0), -1, self.num_features)
```

#### 方案 A2: SNMA 作为"检索增强器"

让 SNMA 生成一个**动态偏移向量**，增强 CHRC 的检索键。

```
Query_enhanced = Query_POGT + γ × SNMA(POGT)
```

**原理**: SNMA 学习当前状态的"上下文信息"，帮助 CHRC 找到更相关的历史误差。

### 2.3 实验设计

| 实验 | 配置 | 预期 |
|------|------|------|
| A1-base | SNMALight + CHRC(P2+P7+P8), β=0.1 | 验证互补性 |
| A1-ablation | β ∈ {0.05, 0.1, 0.2, 0.3} | 找最优权重 |
| A2-query | SNMA 增强检索键 | 验证检索增强 |
| A-control | 仅 CHRC(P2+P7+P8) | 对照基线 |

---

## 3. 方向 B: 替代检索机制

### 3.1 当前检索机制的局限

当前 CHRC 使用**余弦相似度**:
```python
similarities = torch.mm(query_norm, keys_norm.t())  # [batch, n]
```

**局限性**:
1. **语义无关**: 仅衡量向量方向，不考虑"误差预测"任务
2. **固定度量**: 所有维度权重相同，不能学习哪些特征更重要
3. **高相似度聚集**: P0 诊断显示相似度分布 0.97-0.995，区分度低

### 3.2 方案 B1: 学习相似度度量 (Metric Learning)

**思路**: 学习一个变换矩阵 W，使得相似度度量与误差预测对齐。

```python
# 原始: sim = cos(q, k)
# 新: sim = q^T @ W @ k  (双线性)
# 或: sim = MLP([q; k; q-k; q*k])  (非线性)
```

**学习目标**: 最小化检索误差
```
L_retrieval = ||Error_retrieved - Error_true||^2
```

**实现草图**:
```python
class LearnedSimilarity(nn.Module):
    """学习的相似度度量"""

    def __init__(self, feature_dim, method='bilinear'):
        super().__init__()
        self.method = method

        if method == 'bilinear':
            # 双线性变换: q^T W k
            self.W = nn.Parameter(torch.eye(feature_dim))

        elif method == 'mlp':
            # 非线性组合
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim * 4, feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, 1),
            )

    def forward(self, query, keys):
        """
        query: [batch, feature_dim]
        keys: [n, feature_dim]
        returns: [batch, n] similarities
        """
        if self.method == 'bilinear':
            # q^T W k
            transformed = query @ self.W  # [batch, feature_dim]
            return torch.mm(transformed, keys.t())  # [batch, n]

        elif self.method == 'mlp':
            # 对每个 (query, key) 对计算相似度
            batch_size = query.size(0)
            n = keys.size(0)

            q_exp = query.unsqueeze(1).expand(-1, n, -1)  # [batch, n, dim]
            k_exp = keys.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n, dim]

            combined = torch.cat([
                q_exp, k_exp,
                q_exp - k_exp,  # 差异
                q_exp * k_exp,  # 交互
            ], dim=-1)  # [batch, n, dim*4]

            return self.mlp(combined).squeeze(-1)  # [batch, n]
```

### 3.3 方案 B2: 对比学习预训练检索器 (Contrastive Learning)

**思路**: 使用对比学习预训练 POGT 编码器，使得"产生相似误差的 POGT"在特征空间中更接近。

**正负样本定义**:
- **正样本**: 误差相似 (||Error_i - Error_j|| < τ)
- **负样本**: 误差不相似

**损失函数** (InfoNCE):
```python
def contrastive_loss(query, pos_key, neg_keys, temperature=0.1):
    """
    query: 当前 POGT 特征
    pos_key: 误差相似的历史 POGT 特征
    neg_keys: 误差不相似的历史 POGT 特征
    """
    pos_sim = (query * pos_key).sum(dim=-1) / temperature
    neg_sims = torch.mm(query, neg_keys.t()) / temperature

    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
    labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)

    return F.cross_entropy(logits, labels)
```

**训练流程**:
1. 预训练阶段: 用对比学习训练 POGT 编码器
2. 在线阶段: 冻结编码器，使用学习的特征进行检索

### 3.4 方案 B3: 注意力检索 (Attention-based Retrieval)

**思路**: 用 cross-attention 替代 top-K 硬检索，实现软检索。

```python
class AttentionRetrieval(nn.Module):
    """基于注意力的软检索"""

    def __init__(self, feature_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, query, memory_keys, memory_values):
        """
        query: [batch, 1, feature_dim]
        memory_keys: [n, feature_dim]
        memory_values: [n, horizon, num_features]

        returns: [batch, horizon, num_features]
        """
        # 扩展维度
        keys = memory_keys.unsqueeze(0).expand(query.size(0), -1, -1)
        values_flat = memory_values.flatten(1).unsqueeze(0).expand(query.size(0), -1, -1)

        # Cross-attention
        attended, attn_weights = self.attention(query, keys, values_flat)

        return attended.reshape(query.size(0), -1, self.num_features)
```

**优势**:
- 软检索: 不硬选 top-K，而是加权所有历史
- 端到端可学习
- 自然处理多头注意力

**风险**:
- 计算成本高 (O(n) 每次检索)
- 可能过拟合

### 3.5 实验设计

| 实验 | 配置 | 预期 |
|------|------|------|
| B1-bilinear | 双线性相似度 | 快速验证学习度量 |
| B1-mlp | MLP 相似度 | 更强表达力 |
| B2-contrastive | 对比学习预训练 | 更对齐的特征空间 |
| B3-attention | 注意力检索 | 软检索效果 |
| B-control | 余弦相似度 (当前) | 对照基线 |

---

## 4. 组合策略

### 4.1 渐进式验证

**阶段 1**: 分别验证 A 和 B 的有效性
- 如果 A1/A2 有效 → SNMA 可以互补
- 如果 B1/B2/B3 有效 → 检索机制可以改进

**阶段 2**: 组合最佳方案
- A + B 组合
- 与 P2+P7+P8 叠加

### 4.2 优先级排序

基于实现复杂度和预期收益：

| 优先级 | 方案 | 复杂度 | 预期收益 | 理由 |
|--------|------|--------|---------|------|
| **P1** | A1 (SNMALight) | 低 | 中 | 最简单验证 SNMA 是否有互补价值 |
| **P2** | B1-bilinear | 低 | 中 | 最简单的学习相似度 |
| **P3** | B3-attention | 中 | 高 | 端到端可学习，但计算成本 |
| **P4** | B2-contrastive | 高 | 高 | 需要预训练，但原理清晰 |
| P5 | A2 (检索增强) | 中 | 中 | 依赖 B 的效果 |
| P6 | B1-mlp | 中 | 中 | 如果 bilinear 不够表达力 |

---

## 5. 实施规划（Implementation Plan）

### 5.1 目标与边界
- 目标：在 P2+P7+P8 基线之上验证 A/B 方向是否带来持续增益。
- 边界：不修改既有实验记录结论；所有新能力必须有开关，随时可回滚。
- 评估数据集：ETTm1 + Weather。

### 5.2 阶段拆分与顺序
**S0 基线冻结**
1. 固化当前最优配置（P2+P7+P8、bucket=4、softmax）。
2. 建立回滚点（新增功能默认为关闭）。

**S1 方向 A：SNMA-Lite**
1. A1：残差预测分支（不改 backbone、不生成 LoRA）。
2. A1 消融：beta in {0.05, 0.1, 0.2, 0.3}。
3. A2：Query 增强（仅在 A1 有收益时开启）。

**S2 方向 B：检索机制升级**
1. B1-bilinear：学习相似度度量。
2. B3-attention：注意力检索替代 top-k。
3. B2-contrastive：预训练 POGT encoder（后置）。

**S3 组合验证**
- A1 + B1 或 A1 + B3 与基线对比。

### 5.3 代码实施清单（按模块）
**A1: SNMALight**
- 新增模块：`adapter/module/snma_light.py`
- HMem 集成：`adapter/hmem.py`
  - 输出 `delta_snma`，最终预测：`y = y + beta * delta_snma`
- 新参数：
  - `--use_snma_light`（bool）
  - `--snma_beta`（float）
  - `--snma_memory_dim`（int）
- 优化器：`exp/exp_hmem.py` 加入 snma_light 参数组

**A2: Query 增强**
- 在 CHRC 里支持 `query = normalize(pogt_feat + gamma * snma_feat)`
- 新参数：
  - `--chrc_use_snma_query`（bool）
  - `--chrc_snma_query_scale`（float）

**B1: Bilinear Similarity**
- `util/error_bank.py` 新增相似度模式 `bilinear`
- 新参数：`--chrc_similarity_mode=cosine|bilinear`
- 训练方式：W 在在线损失中更新（与 top-k 兼容）

**B3: Attention Retrieval**
- 新增检索器 `AttentionRetrieval`
- 新参数：`--chrc_retrieval_mode=topk|attention`
- 需要 memory sampling/限制规模

**B2: Contrastive Pretrain**
- 新脚本：`scripts/pretrain_pogt_encoder.py`
- 新参数：`--chrc_pretrained_encoder_path`
- 在线阶段冻结编码器

### 5.4 实验节奏
1. **A1-base**（ETTm1/Weather）
2. **A1-beta sweep**
3. **B1-bilinear**（优先 ETTm1）
4. **B3-attention**（可先小规模验证）
5. **A1 + B1 / A1 + B3** 组合对比

### 5.5 成功标准
- ETTm1 MSE < 0.700 或 Weather MSE < 1.45。
- 单次增益 <0.5% 连续两次视为无效。
- 最优方案至少复现 2 次。

### 5.6 风险与应对
- SNMA 冲突：限制 `beta` 并设置 warmup。
- Attention 成本：memory 采样或限制最大容量。
- Bilinear 学习无效：若 top-k 非可微影响过大，改为 soft retrieval 再试。

