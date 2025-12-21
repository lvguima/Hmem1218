# H-Mem 2.0: CHRC-Centric Architecture Redesign

**Date:** 2025-12-20
**Status:** Strategic Pivot & Implementation Plan
**Based on:** Experimental analysis (some_exp_results.md), parameter diagnosis (1220parameter_diagnosis.md), and architecture review

---

## 1. Strategic Decision: Focusing on CHRC

### 1.1 The Rationale for Deprecating SNMA

Recent experiments revealed critical structural weaknesses in the dual-adapter architecture:

| Evidence | Finding |
|----------|---------|
| **Split Brain Conflict** | SNMA + CHRC (MSE 0.814) performs **worse** than either alone (SNMA: 0.776, CHRC: 0.780) |
| **Noise Amplification** | SNMA degrades Weather by +35% vs frozen baseline (2.317 vs 1.714) |
| **ROI Asymmetry** | CHRC provides stable gains with lower complexity; SNMA requires hyper-sensitive tuning (LR 1e-5 → 3e-5 causes +12% MSE) |

**Decision:** Temporarily deprecate SNMA and focus all resources on perfecting **CHRC (Cross-Horizon Retrieval Corrector)** — building a robust "Neural Case-Based Reasoning" system.

### 1.2 Current CHRC Performance

| Dataset | Frozen Baseline | CHRC Only | Improvement |
|---------|-----------------|-----------|-------------|
| ETTm1 | 0.829 | 0.780 | **-5.9%** |
| Weather | 1.714 | 1.578 | **-7.9%** |

CHRC already works. The goal is to make it work **much better**.

---

## 2. Theoretical Framework: Why Current CHRC is Limited

### 2.1 The Naive Assumption

Current CHRC operates on: **"Similar POGT → Similar Errors"**

It retrieves historical errors based solely on POGT embedding similarity.

### 2.2 The Flaw: State Aliasing Problem

In non-stationary time series, the **same POGT pattern** can lead to **different errors** depending on:

1. **The Prediction Itself:** What did the model actually predict this time?
2. **The Global Context:** Are we in a rising trend or mean-reverting regime?
3. **The Temporal Position:** Is this a recurring seasonal pattern or a one-off event?

**Example:**
- Historical: POGT=[1,2,3], Prediction=[4,5,6], Error=[-0.5, -0.3, -0.1]
- Current: POGT=[1,2,3], Prediction=[7,8,9], Error=???

Current CHRC would retrieve the historical error and apply it, but the prediction context is completely different!

### 2.3 The Solution: Context-Aware Retrieval

We must upgrade from **Input-Based Retrieval** to **Context-Aware Retrieval**.

The question changes from:
> "When I saw input X, what was my mistake?"

To:
> "When I saw input X **AND** predicted Y, what was my mistake?"

### 2.4 Causality and Observation Window (Leakage Risk)

**Problem:** In the current data loader, `seq_y` contains *future* horizon values. Using `seq_y` to build POGT for prediction effectively leaks future GT.

**Fix:** Enforce a strictly causal POGT source:
```
pogt = batch_x[:, -pogt_len:, :]  # last observed window only
```
This is weaker supervision but aligns with real online constraints and prevents "open-book" behavior.

---

## 3. Design Proposals (Prioritized)

### 3.1 [P1] Dual-Key Contextual Retrieval — Core Innovation

**Problem:** POGT similarity ≠ Error similarity

**Solution:** Construct composite retrieval key capturing both input (cause) and prediction (intent).

**Mechanism:**
```
Key₁: K_pogt = POGTEncoder(pogt)           # The "Situation"
Key₂: K_pred = PredEncoder(prediction)     # The "Intent"
Fused Query: Q = Linear(Concat(K_pogt, K_pred))
```

**Mathematical Formulation:**
$$Q = W_{fuse} \cdot [K_{pogt} \| K_{pred}] + b_{fuse}$$

**Why It Works:**
- If current prediction differs from historical prediction (even with similar POGT), fused key won't match
- Drastically reduces **Negative Transfer**
- Retrieval becomes "situation + action → outcome" mapping

**Implementation:**
```python
class PredictionEncoder(nn.Module):
    def __init__(self, horizon, num_features, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(horizon * num_features, feature_dim * 2),
            nn.LayerNorm(feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, prediction):
        # prediction: [B, H, F]
        return self.encoder(prediction.flatten(1))  # [B, feature_dim]

class CHRC:
    def __init__(self, ...):
        self.pogt_encoder = POGTFeatureEncoder(...)
        self.pred_encoder = PredictionEncoder(horizon, num_features, feature_dim)
        self.key_fusion = nn.Linear(feature_dim * 2, feature_dim)

    def _compute_fused_key(self, pogt, prediction):
        pogt_feat = self.pogt_encoder(pogt)      # [B, D]
        pred_feat = self.pred_encoder(prediction) # [B, D]
        fused = self.key_fusion(torch.cat([pogt_feat, pred_feat], dim=-1))
        return F.normalize(fused, p=2, dim=-1)    # [B, D]
```

---

### 3.2 [P2] Adaptive Confidence Gating — Soft Calibration

**Problem:** Current `chrc_min_similarity` is a hard threshold → brittle, experiments show no effect (likely because similarity distribution is narrow)

**Solution:** Continuous scaling of correction strength based on retrieval quality.

**Mechanism:**
$$\alpha = \sigma(\gamma \cdot (\text{MaxSim} - \tau))$$
$$\text{Correction}_{final} = \alpha \cdot \text{Correction}_{raw}$$

Where:
- $\text{MaxSim}$: Similarity of top match
- $\tau$: Trust threshold (learnable or fixed, e.g., 0.5)
- $\gamma$: Steepness (e.g., 10.0)

**Why It Works:**
- Smooth transitions: "kind of good" → small correction; "perfect" → full correction; "bad" → near-zero
- Auto-abstention when retrieval quality is low
- No hard boundary effects

**Implementation:**
```python
def _compute_confidence_gate(self, max_similarity):
    tau = self.trust_threshold  # e.g., 0.5
    gamma = self.gate_steepness  # e.g., 10.0
    alpha = torch.sigmoid(gamma * (max_similarity - tau))
    return alpha  # [B, 1]

def forward(self, prediction, pogt):
    ...
    # After retrieval and aggregation
    max_sim = similarities.max(dim=-1, keepdim=True)[0]
    alpha = self._compute_confidence_gate(max_sim)

    corrected = prediction + alpha.unsqueeze(-1) * correction
    return corrected
```

---

### 3.3 [P3] Trajectory-Constraint Retrieval — Anti-Noise

**Problem:** Current CHRC treats each step as independent query → susceptible to transient noise matches

**Solution:** Leverage temporal continuity: if step t matches historical k, then t+1 should prefer k+1.

**Mechanism:**
$$S_i = \text{Similarity}(Q, K_i) + \beta \cdot \mathbb{I}(i \in \text{Successors}(\text{LastTopK}))$$

Where:
- $\beta$: Trajectory bias strength (e.g., 0.1-0.3)
- $\text{Successors}(\text{LastTopK})$: Indices {k+1} for each k in last step's top-K

**Why It Works:**
- Encourages "locking on" to continuous historical segments
- Filters random noise matches (true patterns are continuous)
- Especially effective for **periodic data** (weekly/daily cycles)

**Implementation:**
```python
class CHRC:
    def __init__(self, ...):
        ...
        self.last_topk_indices = None
        self.trajectory_bias = 0.2

    def retrieve(self, query, top_k):
        # Base similarity
        sims = F.cosine_similarity(
            query.unsqueeze(1),  # [B, 1, D]
            self.memory_bank.keys[:self.memory_bank.current_size].unsqueeze(0),  # [1, N, D]
            dim=-1
        )  # [B, N]

        # Apply trajectory bias
        if self.last_topk_indices is not None:
            successor_mask = torch.zeros_like(sims)
            for b in range(sims.size(0)):
                for idx in self.last_topk_indices[b]:
                    next_idx = idx + 1
                    if next_idx < sims.size(1):
                        successor_mask[b, next_idx] = self.trajectory_bias
            sims = sims + successor_mask

        # Top-K selection
        top_sims, top_indices = sims.topk(top_k, dim=-1)
        self.last_topk_indices = top_indices.detach()

        return top_sims, top_indices

    def reset_trajectory(self):
        """Call at sequence boundaries"""
        self.last_topk_indices = None
```

---

### 3.4 [P4] Adaptive Aggregation Strategy — Auto-Select

**Problem:** `softmax` vs `weighted_mean` requires manual selection per dataset

**Solution:** Automatically choose based on retrieval similarity distribution.

**Mechanism:**
```
concentration = max_sim / (std_sim + ε)

If concentration HIGH → softmax (clear winner exists)
If concentration LOW → weighted_mean (uncertain, take consensus)
```

**Implementation:**
```python
def adaptive_aggregate(self, retrieved, similarities, valid_mask):
    # Distribution features
    sim_max = similarities.max(dim=-1, keepdim=True)[0]
    sim_std = similarities.std(dim=-1, keepdim=True)
    concentration = sim_max / (sim_std + 1e-8)

    # Soft selection
    softmax_weight = torch.sigmoid(concentration - 5.0)  # threshold tunable

    # Both aggregations
    agg_softmax = self._softmax_aggregate(retrieved, similarities, valid_mask)
    agg_mean = self._weighted_mean_aggregate(retrieved, similarities, valid_mask)

    # Blend
    return softmax_weight.unsqueeze(-1) * agg_softmax + \
           (1 - softmax_weight.unsqueeze(-1)) * agg_mean
```

---

### 3.5 [P5] Error Decomposition Storage – Quality Over Quantity

**Problem:** Storing raw errors mixes systematic bias with random noise

**Solution:** Decompose errors and store only the learnable/systematic component.

**Mechanism:**
```
error = systematic_bias + random_noise
systematic_bias ≈ EMA(recent_errors)  # Smooth, learnable
random_noise = error - systematic_bias  # High-freq, not learnable
```

**Implementation Sketch:**
```python
def store_error(self, pogt, prediction, error):
    fused_key = self._compute_fused_key(pogt, prediction)

    # Decompose error
    if hasattr(self, 'error_ema') and self.error_ema is not None:
        systematic = self.error_ema_decay * self.error_ema + \
                    (1 - self.error_ema_decay) * error
        self.error_ema = systematic.detach()
    else:
        systematic = error
        self.error_ema = error.detach()

    # Store systematic component only
    self.memory_bank.store(fused_key, systematic)
```

---

### 3.6 [P6] Context + Backbone Feature Keys

**Problem:** POGT-only keys remain noise-sensitive and can misalign across regimes.

**Solution:** Use a causal context embedding as part of the key:
```
K_ctx = Encoder(x_tail)  # backbone encoder or lightweight MLP
Q = Fuse(K_pogt, K_pred, K_ctx)
```

**Why It Helps:** Backbone features are smoother and encode regime/seasonal context, improving retrieval stability.

---

### 3.7 [P7] Horizon-Aware Correction Mask

**Problem:** Applying full-horizon corrections can amplify noise at long horizons.

**Solution:** Apply a per-horizon weight:
```
correction[h] *= w[h]   # e.g., decay with horizon or learned mask
```

**Why It Helps:** Allows CHRC to focus on near-term corrections where signal is stronger.

---

### 3.8 [P8] Season/Regime-Aware Memory Buckets

**Problem:** A single memory bank mixes incompatible regimes (day/night, weekday/weekend, seasonal).

**Solution:** Partition memory by time features or detected regimes:
```
bucket_id = f(time_of_day, day_of_week, season)
memory_bank[bucket_id].store(...)
```

**Why It Helps:** Reduces negative transfer across regimes with similar POGT but different dynamics.

---

## 4. Implementation Roadmap

### Phase 0: Foundation (Bug Fix & Baseline) — **Immediate**

**Goal:** Ensure current infrastructure works correctly.

**Tasks:**
1. **Debug `chrc_min_similarity`**
   - Add logging to `ErrorMemoryBank.retrieve()`:
     ```python
     print(f"[CHRC] similarities: min={top_sims.min():.3f}, max={top_sims.max():.3f}, "
           f"mean={top_sims.mean():.3f}, valid_ratio={valid_mask.float().mean():.3f}")
     ```
   - Determine why threshold has no effect (distribution too narrow? effective_confidence too low?)

2. **Clean up SNMA code path**
   - Add `--use_snma False` as default
   - Simplify `HMem.forward()` to skip SNMA entirely when disabled

3. **Establish Golden Baseline**
   ```bash
   # ETTm1 CHRC-only baseline
   python -u run.py --dataset ETTm1 --border_type online --model iTransformer \
     --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
     --only_test --pretrain --online_learning_rate 1e-5 \
     --use_snma False --use_chrc True

   # Weather CHRC-only baseline
   python -u run.py --dataset Weather --border_type online --model iTransformer \
     --seq_len 512 --pred_len 96 --itr 1 --online_method HMem \
     --only_test --pretrain --online_learning_rate 5e-6 \
     --use_snma False --use_chrc True --chrc_aggregation weighted_mean
   ```

**Deliverable:** Confirmed working baseline with diagnostic logging.

4. **Enforce Causal POGT Source**
   - Ensure POGT for prediction uses `batch_x` tail, not `seq_y`
   - Add a flag to choose POGT source for ablation

---

### Phase 1: Dual-Key Contextual Retrieval — **Core Upgrade**

**Goal:** Solve State Aliasing problem.

**Tasks:**
1. Implement `PredictionEncoder` in `util/error_bank.py`
2. Add `key_fusion` layer to `CHRC.__init__()`
3. Modify `CHRC.forward()` to compute fused key
4. Modify `CHRC.store_error()` to store fused key
5. Add flag `--chrc_use_dual_key` (default: True)

**Experiment:**
```bash
# Compare POGT-only vs Dual-Key
python -u run.py ... --chrc_use_dual_key False  # Baseline
python -u run.py ... --chrc_use_dual_key True   # New
```

**Expected Outcome:**
- Weather: Significant improvement (reduced negative transfer)
- ETTm1: Comparable or slight improvement

---

### Phase 2: Adaptive Confidence Gating — **Stability**

**Goal:** Replace brittle hard threshold with smooth gating.

**Tasks:**
1. Add `trust_threshold` and `gate_steepness` params to CHRC
2. Replace `valid_mask` logic with soft alpha gating
3. Remove or deprecate `chrc_min_similarity`

**Experiment:**
```bash
# Sweep trust_threshold
python -u run.py ... --chrc_trust_threshold 0.3
python -u run.py ... --chrc_trust_threshold 0.5
python -u run.py ... --chrc_trust_threshold 0.7
```

**Expected Outcome:**
- More stable performance across threshold values
- Automatic abstention on OOD samples

---

### Phase 3: Trajectory Constraint — **Polish**

**Goal:** Improve noise robustness on periodic data.

**Tasks:**
1. Add `last_topk_indices` state to CHRC
2. Implement successor bias in `retrieve()`
3. Add `reset_trajectory()` for sequence boundaries
4. Add flag `--chrc_trajectory_bias` (default: 0.2)

**Experiment:**
```bash
# Compare with/without trajectory
python -u run.py ... --chrc_trajectory_bias 0.0  # Disabled
python -u run.py ... --chrc_trajectory_bias 0.2  # Enabled
```

**Expected Outcome:**
- ETTm1 (periodic): Notable improvement
- Weather (noisy): Slight improvement or neutral

---

### Phase 4: Optional Enhancements

**P4: Adaptive Aggregation** — If softmax vs weighted_mean remains dataset-dependent
**P5: Error Decomposition** — If noise in stored errors is identified as issue

---

## 5. Architecture 2.0 Summary

```
┌─────────────────────────────────────────────────────────────┐
│                      H-Mem 2.0 (CHRC-Only)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: X_history ──► Frozen Backbone ──► Ŷ_base           │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    CHRC Module                       │   │
│  │                                                      │   │
│  │  POGT ──► POGTEncoder ──┐                           │   │
│  │                         ├──► KeyFusion ──► Q        │   │
│  │  Ŷ_base ──► PredEncoder ─┘                          │   │
│  │                                                      │   │
│  │  Q ──► MemoryBank.retrieve() ──► TopK (errors, sims)│   │
│  │           │                                          │   │
│  │           ├── Trajectory Bias (optional)             │   │
│  │           │                                          │   │
│  │  TopK ──► AdaptiveAggregate() ──► Δ (correction)    │   │
│  │                                                      │   │
│  │  MaxSim ──► SoftGate(τ, γ) ──► α                    │   │
│  │                                                      │   │
│  │  Ŷ_final = Ŷ_base + α · Δ                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [After H steps: GT revealed]                               │
│  Error = GT - Ŷ_final                                       │
│  MemoryBank.store(Q, Error)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovations:**
1. **Dual-Key Retrieval:** Q = Fuse(POGT, Prediction) — solves State Aliasing
2. **Soft Gating:** α = σ(γ·(MaxSim - τ)) — replaces hard threshold
3. **Trajectory Bias:** S_i += β if i ∈ Successors(LastTopK) — enforces temporal consistency
4. **Adaptive Aggregation:** Auto-select softmax/weighted_mean based on similarity distribution

---

## 6. Success Metrics

| Metric | Current | Target (Phase 1) | Target (Full) |
|--------|---------|------------------|---------------|
| ETTm1 MSE | 0.780 | ≤ 0.770 | ≤ 0.750 |
| Weather MSE | 1.578 | ≤ 1.500 | ≤ 1.400 |
| Cross-dataset variance | High | Medium | Low |
| Parameter sensitivity | High | Medium | Low |

---

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Dual-Key increases storage cost | Keep feature_dim same; fused key replaces POGT key |
| Trajectory bias causes "stuck" retrieval | Add decay to bias; reset at sequence boundaries |
| Soft gating learns to always abstain | Initialize τ low; monitor α distribution |
| PredEncoder adds latency | Use lightweight MLP; batch encode |

---

## 8. Next Steps

1. **Today:** Implement Phase 0 diagnostics, establish baseline
2. **Tomorrow:** Implement Dual-Key (Phase 1)
3. **Day 3:** Run comparison experiments
4. **Day 4-5:** Phase 2-3 based on results
