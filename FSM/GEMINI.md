# 📚 Online Time Series Forecasting & Continual Learning Knowledge Base

**Date:** 2025-12-12
**Source Material:** 11 Academic Papers (2020-2025) covering Online Learning, Time Series Forecasting (TSF), and Continual Learning (CL).

---

## 1. 🧠 Core Domain Overview
**Online Time Series Forecasting** addresses the challenge of predicting future values based on streaming data where the data distribution changes over time (**Non-stationarity** / **Concept Drift**). Unlike offline batch training, models must:
1.  **Adapt Quickly (Plasticity):** Learn new patterns immediately as data arrives.
2.  **Retain Knowledge (Stability):** Avoid **Catastrophic Forgetting** of historical patterns (e.g., recurring seasonalities).
3.  **Handle Feedback Delay:** Ground truth is often not available immediately after prediction, creating a temporal gap that standard online learning ignores.

---

## 2. 🔬 Taxonomy of Approaches

The analyzed literature tackles these challenges through several primary mechanisms:

### A. Architecture-Based Adaptation (Fast Weights & Ensembling)
Instead of simple Gradient Descent (GD) updates, these methods modify the model architecture or weights dynamically.
*   **FSNet (Pham et al., 2022):** Inspired by Complementary Learning Systems (CLS). Uses a standard backbone (Slow weights) and per-layer **Adapters** (Fast weights) driven by gradient EMA. Includes an **Associative Memory** to recall recurring concept drifts.
*   **OneNet (Zhang et al., 2023):** Argues that no single model structure (Cross-Time vs. Cross-Variable) works for all drifts. Uses **Online Ensembling** with Reinforcement Learning (RL) to dynamically weight two branches.
*   **Titans (Behrouz et al., 2024):** Introduces a **Neural Long-term Memory** module. Unlike static weights, this memory module *learns* to memorize historical context at test time via gradient descent on a "surprise" metric, effectively acting as a meta-learner within the architecture.

### B. Proactive & Test-Time Adaptation (Handling Feedback Delay)
Addressing the "Temporal Gap" where ground truth is delayed.
*   **Proceed (Zhao & Shen, 2025):** Estimates **Concept Drift** between training and test data in a latent space. Uses a Hypernetwork to generate parameter shifts *before* prediction.
*   **TAFAS (Kim et al., 2025):** Exploits **Partially-Observed Ground Truth (POGT)**. Since TSF is sequential, early time steps of the ground truth become available before the full horizon. It uses Periodicity-Aware Scheduling (PAAS) to use this partial data for immediate adaptation.

### C. Disentanglement & Causal Mechanisms
Moving beyond statistical correlation to understand the *source* of drift.
*   **LSTD (Cai et al., 2025):** Disentangles **Long-term stable states** (e.g., economic cycles) from **Short-term intervention states** (e.g., sudden policies). Uses "smooth" and "interrupted dependency" constraints to separate these latent variables.

### D. Replay & Memory Buffers
Storing past data to prevent forgetting.
*   **DER/DER++ (Buzzega et al., 2020):** Stores network **Logits** (soft targets) alongside raw data to preserve "functional" knowledge.
*   **Adaptive CL for Industrial TSF (Wu et al., 2024):** Uses a **Dual-Buffer** (Standard + Soft Buffer for hard samples) and **Hint-based learning** to distill intermediate feature maps.

### E. Benchmarking & Evaluation (Class-Incremental)
*   **TSCIL Benchmark (Qiao et al., 2024):** A comprehensive evaluation of Class-Incremental Learning for Time Series. Establishes that **LayerNorm** significantly outperforms BatchNorm in CL settings and that **Replay** methods generally beat regularization methods in this domain.

---

## 3. 📄 Detailed Paper Summaries

### 1. Learning Fast and Slow for Online Time Series Forecasting (FSNet)
*   **Authors:** Pham et al. (2022)
*   **Mechanism:** **Adapters** (Fast weights) + **Associative Memory** (Recurring patterns).
*   **Verdict:** The de facto SOTA baseline. Excellent for Recurring Concept Drift.

### 2. OneNet: Enhancing TSF Models under Concept Drift by Online Ensembling
*   **Authors:** Zhang et al. (2023)
*   **Mechanism:** Dynamic Ensembling of **Cross-Time** and **Cross-Variable** models using **Online Convex Programming (OCP)**.
*   **Verdict:** Reduces error by >50% vs SOTA. Proves ensembling is superior to single-model adaptation.

### 3. Proactive Model Adaptation Against Concept Drift (Proceed)
*   **Authors:** Zhao & Shen (2025)
*   **Mechanism:** **Latent Drift Estimation** + **Hypernetwork Generator**. Predicts parameter updates *before* the ground truth arrives.
*   **Verdict:** Explicitly tackles the **Feedback Delay** issue.

### 4. Battling Non-stationarity via Test-time Adaptation (TAFAS)
*   **Authors:** Kim et al. (2025)
*   **Mechanism:** **Partially-Observed Ground Truth (POGT)**. Uses the first few arriving data points of the horizon to adapt the model immediately via **Gated Calibration Modules**.
*   **Verdict:** A practical "Test-Time Adaptation" approach that turns the sequential nature of TS into an advantage.

### 5. Titans: Learning to Memorize at Test Time
*   **Authors:** Behrouz et al. (2024)
*   **Mechanism:** **Neural Long-term Memory**. A memory module that updates its own weights at test time based on a "surprise" loss (gradient-based).
*   **Verdict:** A new architecture class. Scales better than Transformers (2M+ context) by compressing history into learnable weights.

### 6. Disentangling Long-Short Term State (LSTD)
*   **Authors:** Cai et al. (2025)
*   **Mechanism:** **Causal Disentanglement**. Separates stable long-term states from intervention-based short-term states.
*   **Verdict:** Theoretically grounded; essential for shifts caused by specific "interventions".

### 7. Adaptive CL Method for Nonstationary Industrial TSF
*   **Authors:** Wu et al. (2024)
*   **Mechanism:** **Dual-Buffer** + **Hint-based Learning** + **TimeRelu**.
*   **Verdict:** Optimized for industrial stability.

### 8. Class-incremental Learning for Time Series (TSCIL)
*   **Authors:** Qiao et al. (2024)
*   **Key Findings:**
    *   **LayerNorm > BatchNorm** for Continual Learning.
    *   **Replay > Regularization** for Time Series.
    *   **Intra-class variation** (e.g., different subjects) is a major hurdle.
*   **Verdict:** Establishes the standard benchmark for classification-based TS tasks.

### 9. Dark Experience for General Continual Learning (DER/DER++)
*   **Authors:** Buzzega et al. (2020)
*   **Mechanism:** Replay with **Logit Distillation**.
*   **Verdict:** The foundational baseline for modern Replay methods.

### 10. CLeaR: Adaptive CL Framework for Regression
*   **Authors:** He & Sick (2021)
*   **Mechanism:** **Novelty Detection** (Autoencoder) triggers updates via **Online EWC**.
*   **Verdict:** Modular framework for regression.

### 11. Rethinking Momentum Knowledge Distillation (MKD)
*   **Authors:** Michel et al. (2024)
*   **Mechanism:** **EMA Teacher** for consistent distillation targets in online streams.
*   **Verdict:** Solves "Teacher Quality" instability in online learning.

---

## 4. 🚀 New Proposed Methodology: H-Mem

Based on the synthesis of the above research, a new method **H-Mem (Horizon-Bridging Neural Memory Network)** is proposed.

*   **Core Innovation:** Bridges the "Feedback Delay" gap by combining **Neural Memory** (for immediate short-term adaptation using Partial Ground Truth) and **Retrieval Augmented Generation** (for long-term correction using historical residuals).
*   **Components:**
    1.  **Frozen Backbone:** Ensures stability.
    2.  **Short-term Neural Memory Adapter (SNMA):** Learns from POGT via Hypernetworks (inspired by Titans/FSNet) to handle covariate shift.
    3.  **Cross-Horizon Retrieval Corrector (CHRC):** Retrieves historical error patterns to correct the final prediction (inspired by Proceed/TAFAS).
*   **Status:** Proposed Design (See `H-mem.md` for details).

---

## 5. 🔗 Comparative Analysis & Trends

| Feature | FSNet (2022) | OneNet (2023) | Proceed (2025) | TAFAS (2025) | Titans (2024) | **H-Mem (New)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Core Idea** | Fast/Slow Weights | Ensemble | Proactive Hypernet | Partial Ground Truth | Neural Memory | **Neural Memory + RAG** |
| **Adaptation** | Gradient on Adapter | RL Weighting | Predicted Shift | Gradient on Gating | Gradient on Memory | **Hypernet LoRA + Retrieval** |
| **Drift Handling**| Reactive | Reactive | **Proactive** | **Proactive** (via POGT) | In-Context Learning | **Proactive & Corrective** |
| **Memory** | Associative | None | None | None | **Neural Weights** | **Dual (Neural + Database)** |

**Emerging Trend:**
1.  **Proactive Adaptation:** Moving from reacting to errors (FSNet/OneNet) to predicting shifts (Proceed) or using early signals (TAFAS) to handle the **Feedback Delay**.
2.  **Neural Memory:** Moving from raw data buffers (DER) to compressing history into model weights (Titans/LSTD).
3.  **Standardization:** The field is maturing with rigorous benchmarks (TSCIL) clarifying best practices (LayerNorm, Replay).
