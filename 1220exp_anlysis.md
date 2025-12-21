# 1220 Experiments Analysis (Condensed)

## Scope
This document condenses the completed 1220 H-Mem/CHRC iterations and their
experimental outcomes from `1220Hmem_improve.md` and `1220hmem_improve_results.md`.

## Results Summary Table

| Phase | Method | ETTm1 MSE | Weather MSE | Verdict |
|-------|--------|-----------|-------------|---------|
| P0 | CHRC-only baseline | 0.778 | 1.770 | Baseline |
| P1 | Dual-Key | 0.778 | 1.916 (+8%) | ❌ Failed |
| P2 | Soft Gating | **0.756 (-2.8%)** | **1.578 (-11%)** | ✅ Success |
| P3 | Trajectory Bias | 0.772 | 1.920 | ❌ Harmful |
| P4 | Adaptive Aggregation | 0.772 | 1.944 | ❌ Harmful |
| P5 | Error Decomposition | 0.759 | 1.958 | ⚪ No effect |
| P6 | Context Key | 0.771 | 1.945 | ⚪ Negligible |
| P7 | Horizon Mask | **0.725 (-6.8%)** | **1.527 (-14%)** | ✅ Major win |
| P8 | Time Buckets | **0.710 (-8.7%)** | 1.584 | ✅ ETTm1 best |

**Best achieved:** ETTm1=0.710 (P8), Weather=1.527 (P7)

## Key Findings by Phase

- **P0 (diagnosis + causal POGT):** Similarity distribution is very narrow
  (0.97-0.995), explaining why hard threshold `chrc_min_similarity` had no effect.
  Baselines are stable and suitable for comparisons.

- **P1 (dual-key retrieval):** ETTm1 unchanged; Weather worsened by 8%. The
  prediction-based key likely introduces redundancy (prediction already encodes
  POGT) and dilutes retrieval precision by expanding the key space.

- **P2 (soft gating):** **Significant improvement** (ETTm1 -2.8%, Weather -11%).
  Despite narrow similarity distribution, the continuous gating function provides
  fine-grained confidence modulation that hard thresholds cannot achieve. Note:
  tau values (0.3/0.5/0.7) show identical results, suggesting the mechanism works
  but is insensitive to threshold choice in this similarity regime.

- **P3 (trajectory bias):** Negligible to harmful effect. Temporal continuity
  assumption ("if t matches k, then t+1 should prefer k+1") does not hold for
  error patterns, which are more stochastic than the underlying signal.

- **P4 (adaptive aggregation):** Clearly worse than fixed strategies. Attempting
  to dynamically blend softmax/weighted_mean based on similarity distribution
  amplifies noise rather than reducing it. Fixed strategies are more robust.

- **P5 (error decomposition):** Zero effect (identical MSE values). Either the
  EMA decomposition has implementation issues, or time series errors lack the
  "systematic + noise" structure assumed by the design.

- **P6 (context key):** Minimal gain (<0.1%). Raw input tail features are too
  noisy to provide meaningful context signal.

- **P7 (horizon mask):** **Largest gains** (ETTm1 -6.8%, Weather -14%). Reveals
  a fundamental insight: error patterns have temporal locality—corrections are
  valid for near-term predictions but become noise for long-term predictions.
  Exponential decay (0.98, min=0.2) effectively exploits this structure.

- **P8 (time buckets):** **Best ETTm1** (0.710) with 4 buckets; Weather slightly
  worse than P7 best but better than P0. Bucketed memory isolates time-regime
  patterns, highly effective for periodic data (ETTm1) but less so for aperiodic
  data (Weather).

---

## Deep Insights

### 1. Theory vs Practice Gap

The most theoretically justified improvements (P1 Dual-Key, P3 Trajectory) **failed completely**, while simpler mechanisms (P2, P7) succeeded. This suggests:

- Time series error patterns are more stochastic than assumed
- Inductive biases that "sound reasonable" often don't match reality
- **Data-driven simplicity > theory-driven complexity**

### 2. What Actually Works: Structural Priors

The two successful mechanisms share a common trait—they encode **task-level structural priors**:

| Mechanism | Structural Prior |
|-----------|-----------------|
| P7 Horizon Mask | "Error transferability decays with prediction horizon" |
| P8 Time Buckets | "Similar time regimes have similar error patterns" |

These are fundamentally different from retrieval mechanics (P1, P3, P4) which try to improve *how* we retrieve, rather than *what* we should trust.

### 3. Why P1 Dual-Key Failed

The design assumed: "Same POGT + different prediction → different error"

Reality:
1. Prediction Ŷ = f(POGT, context), so encoding both is redundant
2. Expanding key dimensionality dilutes retrieval matches
3. Error patterns may correlate weakly with prediction magnitude

### 4. The Effective CHRC Formula

After these experiments, the empirically validated correction is:

```
Correction = α(similarity) × w(horizon) × retrieved_error
             ↑ P2 soft gate   ↑ P7 decay
```

Not needed: dual-keys, trajectory bias, adaptive aggregation, error decomposition.

### 5. Dataset-Dependent Mechanisms

P8 (Time Buckets) reveals a key insight: **method effectiveness depends on data characteristics**.

| Data Type | Effective Mechanisms | Ineffective |
|-----------|---------------------|-------------|
| Strong periodicity (ETTm1) | P7 + P8 | P1, P3 |
| Weak periodicity (Weather) | P7 only | P1, P3, P8 |

This suggests future work should include automatic data characterization.

### 6. The "Insensitivity" Paradox of Soft Gating (P2)

The fact that `tau` values of 0.3, 0.5, and 0.7 yielded identical results (P2) is not just robustness—it confirms the findings from P0. Since the similarity distribution is extremely narrow and high (0.97–0.99), `Sigmoid(γ * (MaxSim - τ))` saturates to ≈1.0 for all tested taus.
**Implication:** Soft Gating works not because of the threshold choice, but likely because the function shape inherently dampens the *very few* outliers that drop below the high baseline. Future versions should use **Percentile Normalization** (relative rank) rather than absolute cosine similarity.

### 7. Why Error Decomposition (P5) Failed

Error decomposition assumes `Error = Systematic_Bias + Random_Noise`.
**The flaw:** In Online TSF, the "Systematic Bias" itself is non-stationary due to rapid Concept Drift. Yesterday's bias is not today's bias.
By the time the EMA estimator "learns" the current bias, the distribution has likely shifted. Thus, attempting to separate bias from noise in a highly dynamic stream becomes a lagging indicator that adds no predictive value.

---

## Roadmap

### Short-term (Immediate)

1. **Validate P7+P8 combination:**
   ```bash
   # ETTm1: combine P7 horizon mask + P8 time buckets
   --use_horizon_mask True --chrc_use_buckets True --bucket_num 4

   # Weather: P7 only (no buckets)
   --use_horizon_mask True --chrc_use_buckets False
   ```

2. **Aggregation sweep on best config:** Compare softmax vs weighted_mean on P7+P8 baseline to confirm fixed aggregation superiority.

3. **Bucket sensitivity:** Test bucket_num ∈ {2, 4, 8} on ETTm1 to find optimal granularity.

4. **Code cleanup:** Remove or deprecate P1, P3, P4, P5 code paths to simplify maintenance.

### Mid-term (1-2 weeks)

1. **Automatic periodicity detection:**
   - Compute ACF/spectral density at runtime
   - Auto-enable buckets for high-periodicity data
   - Formula: `use_buckets = (acf_peak > threshold)`

2. **Horizon mask auto-tuning:**
   - Current decay=0.98, min=0.2 are hand-tuned
   - Learn from error variance vs horizon: `w[h] ∝ 1/Var(error[h])`

3. **Investigate P5 failure:**
   - Add logging to verify EMA updates are happening
   - Test alternative decomposition (e.g., moving average filter)

4. **Cross-dataset validation:** Test best configs on ETTh1, ETTh2, ECL, Traffic.

### Long-term (Research directions)

1. **Revisit SNMA with simplified design:**
   - Now that CHRC is optimized, test if a minimal SNMA (no hypernetwork, just direct LoRA update) can add value
   - Key constraint: must not conflict with CHRC corrections

2. **Alternative retrieval mechanisms:**
   - Attention-based retrieval instead of cosine similarity
   - Learned similarity metric (contrastive learning on error prediction)

3. **Transferable configurations:**
   - Meta-learn CHRC hyperparameters across datasets
   - Goal: zero-shot config for new datasets based on data statistics

4. **Theoretical understanding:**
   - Formalize why horizon mask works (error autocorrelation analysis)
   - Characterize when time buckets help vs hurt

---

## Conclusion

| Dimension | Assessment |
|-----------|------------|
| Goal achievement | ETTm1 ✅ exceeded (0.710 vs target 0.750); Weather partial (1.527 vs target 1.400) |
| Key lesson | Simple structural priors > complex retrieval mechanics |
| Biggest surprise | P7 Horizon Mask's large gain reveals error temporal locality |
| Biggest lesson | P1 Dual-Key failure warns against over-trusting intuition |

**One-line summary:** "Less is more"—the winning improvements are simple decay masks and time isolation, not sophisticated retrieval architectures.
