# H-Mem Implementation Plan (Priority Order)

**Date:** 2025-12-19  
**Scope:** Turn the design issues in `Hmem_design_improve.md` into a concrete, staged refactor plan.

## Prioritization Principles
1) Correctness and causality first (avoid leakage / misalignment).  
2) Stability next (reduce high-variance updates).  
3) Effectiveness and generalization after (avoid negative transfer).  
4) Complexity last (touching multiple modules only when earlier fixes settle).

## Implementation Plan (Step-by-step)

### 1) POGT Observability Alignment (P0)
**Why:** Causality risk and evaluation optimism.  
**Work:**
- Ensure POGT used for prediction is strictly from *observed past* (not from current window GT).
- Rewire `Exp_HMem.online()` to construct POGT from `recent_batch` (or a cached past window) instead of `current_batch`.
- In `_update_online()`, keep POGT from the observed window only.
- Add a lightweight sanity check on POGT indices/length.
**Done when:** Prediction path never touches future GT; unit sanity print/check passes.

### 2) Preserve SNMA Memory Across Steps (P0)
**Why:** Current per-step reset removes temporal smoothing and increases noise chasing.  
**Work:**
- Add `detach_state()` in SNMA/NeuralMemoryState to truncate graphs without wiping memory.
- Replace per-step reset with `detach_state()`; keep full reset only at phase boundaries.
- Add optional `hmem_reset_every` for safety (default off).
**Done when:** SNMA memory persists across steps without graph errors.

### 3) CHRC Abstention on Low Absolute Similarity (P1)
**Why:** Prevent negative transfer when retrieval is weak or OOD.  
**Work:**
- Add `chrc_min_similarity` (absolute threshold).
- Gate correction by max similarity and retrieval quality.
- Default to “no correction” if similarity below threshold.
**Done when:** CHRC correction is zeroed for low-similarity queries.

### 4) Reduce HyperNetwork Volatility (P1)
**Why:** Over-sensitive LoRA params hurt stability.  
**Work:**
- Add EMA smoothing for generated LoRA parameters (or memory state).
- Allow module-wise target selection / reduced LoRA injection surface.
- Optional clamp for generated LoRA scales.
**Done when:** Step-to-step LoRA deltas are bounded and less oscillatory.

### 5) Share / Align POGT Representations (P2)
**Why:** SNMA + CHRC share same signal but encode it independently.  
**Work:**
- Add shared POGT encoder or a projection head reused by both modules.
- Keep backward compatibility via a flag.
**Done when:** Both systems consume the same POGT embedding.

### 6) Confidence Gate Input Simplification (P2)
**Why:** High-dimensional flattened inputs overfit in batch_size=1.  
**Work:**
- Replace flattened tensors with summary stats (mean/std/energy).
- Keep feature_dim constant to avoid retraining large heads.
**Done when:** Gate input dimension shrinks substantially.

### 7) Active Forgetting in Memory Bank (P3)
**Why:** Remove stale patterns in long streams.  
**Work:**
- Track correction effectiveness; decay or evict entries that increase error.
- Add a “stale penalty” term in eviction scores.
**Done when:** Memory bank can suppress outdated patterns.

## Validation Checklist
- **ETTm1** and **Weather** sanity runs after each P0/P1 change.
- Track MSE/MAE + CHRC usage rate + similarity distributions.
- Ensure no runtime NaN/Inf and no autograd graph explosion.
