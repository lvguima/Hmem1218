# Fig Prompts (Section II / Methodology)

## Prompt 1 — New Fig.1 (Overall Architecture)

Create a clean IEEE-paper style vector architecture diagram (white background, minimal colors, crisp lines, high resolution). Font: Times New Roman, bold for module titles, regular for annotations. Use LaTeX-like math for symbols. Layout: two-stream pipeline with strict causality. No “Dual-Key”, no “POGT”. Method name: R-mem; component: RRC; query: Condition Context Snippet.

Canvas: 16:9 horizontal. Top branch = frozen forecasting backbone; bottom branch = residual-memory retrieval correction; right side = fusion to final forecast; include a delayed-feedback memory-write loop.

(1) Left input:
- Box: X_in (lookback window) with arrow splitting to two branches.

(2) Top branch (Main Branch / Frozen Backbone):
- Large blue block labeled: “Frozen Backbone (e.g., PatchTST)” with a lock icon and text “θ_frozen”.
- Output arrow to the right: “Base Prediction  Ŷ_base (H-step)”.

(3) Bottom branch (Correction Branch / RRC):
- Small box: “Condition Context Snippet Extractor” taking X_in and outputting “G_t (snippet)”.
  - Add subtitle: “proxy snippet from recent observations (causal)”.
- Next box: “Key Encoder f_k(·)” producing “q_t (query key)”.
- Memory bank drawn as a database/stack with partitions:
  - Title: “Residual Memory Bank 𝓜”
  - Inside show “Buckets” (stacked shelves) labeled like “bucket 1, bucket 2, …” (time/regime buckets).
  - Each entry stores: “Key k_i” and “Value E_i (residual trajectory, H×C)”.
- Arrow from q_t to the memory bank: “Search within bucket”.
- From memory bank to a “Top-k Retrieval” node (show k neighbors).
- Then a block: “Similarity + Softmax Weights” producing weighted sum.
- Then a block: “Aggregated Residual Trajectory  Ê_t”.

(4) Horizon-aware conservative correction:
- Block: “Horizon-Aware Mask m[h]” applied to Ê_t, output “Ê_t^masked”.
  - Annotation: “farther horizon → more conservative correction”.
- Block: “Confidence Gating” producing scalar/vector “λ_t” (0–1).
  - Annotation: “low retrieval confidence → fallback to backbone”.

(5) Fusion:
- A ⊕ node combining Ŷ_base and “Δ_t = λ_t ⊙ Ê_t^masked”.
- Output box: “Final Prediction  Ŷ_final = Ŷ_base + Δ_t”.

(6) Delayed error learning loop (strict causality):
- Add a timeline/clock icon near the bottom: “Label arrives after delay”.
- Show that at time t+H, ground truth Y_true becomes available.
- Compute residual trajectory: “E_t = Y_true − Ŷ_base (or Ŷ_final, specify consistently with your method text)”.
- Arrow back into memory bank: “Write: push (q_t, E_t) into bucket; FIFO / pop oldest”.
- Label this loop: “Delayed Error Write (causal)”.

Color palette:
- Backbone = blue, Memory/Retrieval = orange, Mask/Gate = green, Fusion = gray/black.
Style:
- Clean, technical, no gradients, consistent arrow thickness, aligned boxes, minimal text but precise.
Include caption text suggestion in small font: “R-mem: frozen backbone + RRC retrieval correction + delayed residual write”.

## Prompt 2 — New Fig.2 (Core Component / RRC Breakdown)

Create an IEEE-style vector “module breakdown” figure focusing on the core component RRC (Residual Retrieval Corrector). White background, Times New Roman (bold for module titles). Use clear data-flow arrows and math notations. No Dual-Key, no POGT.

Canvas: 4:3 or 16:10. Center a big container box titled “RRC: Residual Retrieval Corrector”. Inside show 4 stages: (A) Query construction, (B) Bucketed retrieval, (C) Horizon-aware masking + refinement, (D) Confidence gating + correction output. Add small callouts explaining industrial rationale (delayed assays, non-stationary regimes, long-horizon uncertainty).

Inputs (left side):
- G_t: “Condition Context Snippet (proxy, causal)”
- Ŷ_base: “Base forecast (H-step)”
- optional compact stats Z_t: “Compact drift/error stats (e.g., recent residual norm / similarity entropy)” (keep generic)

(A) Query Construction:
- Box: “Key Encoder f_k(·)” takes G_t → outputs q_t.
- Optional small note: “captures current operating condition fingerprint”.

(B) Bucketed Retrieval:
- Box: “Bucket Selector b(t)” chooses a bucket index (time/regime bucket).
- Database icon: “Residual Memory 𝓜_b” (only the selected bucket).
- Box: “Similarity s_t = sim(q_t, k_i)” producing similarity vector S_t over candidates.
- “Top-k” selector.
- “Softmax attention weights a_i” and “Weighted sum” → output “Ê_t (retrieved residual trajectory, H×C)”.
- Add a mini visualization of S_t (a small bar vector) labeled “S_t”.

(C) Horizon-aware Masking + Refinement:
- Box: “Horizon-Aware Mask m[h]” takes Ê_t → Ê_t^masked.
  - Show three selectable mask types in a small legend:
    - “Exponential: m[h]=exp(−γh)”
    - “Linear: m[h]=max(0,1−βh)”
    - “Learned: m[h]=σ(w_h)”
  - Add note: “choose ONE type per experiment”.
- Box: “Refinement Net r(·)” takes (Ê_t^masked, Z_t) → outputs “Ê_t^ref”.
  - Note: “denoise / align to current context”.

(D) Quality Estimation + Confidence Gating:
- Box: “Quality Estimator” takes (S_t, Z_t) → outputs quality score g_t (0–1).
  - Example features in tiny text: “top1 gap, entropy, ||Ê||, drift stats”.
- Box: “Similarity Gate α_t” derived from S_t (e.g., max similarity).
- Box: “Stats Gate β_t” derived from Z_t (optional).
- Combine gates in a small node: “λ_t = Gate(α_t, g_t, β_t) ∈ [0,1]”.
- Output correction: “Δ_t = λ_t ⊙ Ê_t^ref”.

Final fusion (right side, outside RRC box):
- “Ŷ_final = Ŷ_base + Δ_t”.

Bottom strip: Delayed memory write (small, as reminder):
- “When Y_true arrives (delay): E_t = Y_true − Ŷ_base; write (q_t, E_t) into bucket; FIFO”.

Industrial rationale callouts (small text bubbles):
- “Delayed assays → learn from delayed errors”
- “Non-stationary regimes → retrieve residual patterns under similar conditions”
- “Farther horizon more uncertain → conservative correction via m[h] and gating”

Color palette:
- Retrieval/memory orange, masking green, gating purple, backbone shown only as Ŷ_base input (gray).
Style: minimal, technical, aligned, readable at single-column print size.

