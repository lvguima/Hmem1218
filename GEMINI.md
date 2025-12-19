# GEMINI Project Context: OnlineTSF

## 1. Project Overview
**OnlineTSF** is a comprehensive framework for **Online Time Series Forecasting (OTSF)**. It addresses the challenge of predicting future values in streaming data where distributions change over time (Concept Drift).

The framework emphasizes:
- **No Information Leakage:** Strictly adhering to causal constraints where ground truth is only available *after* the forecast horizon.
- **Fair Comparisons:** Standardized implementations of SOTA methods.
- **New Methodology (H-Mem):** A novel "Horizon-Bridging Neural Memory Network" currently under development in this codebase.

## 2. Environment Setup
The project uses Python and standard deep learning libraries.

### Installation
```bash
pip install -r requirements.txt
```
**Key Dependencies:**
- `torch>=1.12.0`
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `transformers` (for some backbone models)

## 3. Project Structure

### Core Directories
- **`adapter/`**: Contains the implementation of online adaptation modules.
    - `hmem.py`: Main implementation of the **H-Mem** method.
    - `proceed.py`: Implementation of the **Proceed** method.
    - `module/`: Sub-components like `neural_memory.py`, `lora.py`, `generator.py`.
- **`exp/`**: Experiment orchestration logic.
    - `exp_main.py`: Main entry point logic.
    - `exp_online.py`: Base class for online learning experiments.
    - `exp_hmem.py`: Specialized experiment loop for H-Mem.
- **`models/`**: Backbone forecasting models (e.g., `iTransformer`, `PatchTST`, `FSNet`).
- **`layers/`**: Neural network layers (Attention, Embeddings, RevIN).
- **`scripts/`**: Shell/PowerShell scripts for reproducing experiments.
    - `online/`: Scripts for online learning benchmarks.
    - `pretrain/`: Scripts for pre-training backbones (required before online phase).
- **`FSM/`**: Knowledge base and documentation for the H-Mem method.

### Entry Points
- **`run.py`**: The primary CLI entry point for running experiments.
- **`run_all_experiments.py`**: Utility for batch execution.

## 4. Usage & Workflows

### Standard Workflow
1.  **Pre-train** a backbone model (optional but recommended for some methods):
    ```bash
    bash scripts/pretrain/iTransformer/ETTh2.sh
    ```
2.  **Run Online Learning**:
    ```bash
    bash scripts/online/iTransformer/OneNet/ETTh2.sh
    ```

### Running H-Mem (New Method)
H-Mem uses specific flags (`--online_method HMem`) and components (`SNMA`, `CHRC`).

**Example Command:**
```bash
python -u run.py \
  --dataset ETTh2 \
  --model iTransformer \
  --pred_len 96 \
  --online_method HMem \
  --learning_rate 0.0001 \
  --online_learning_rate 0.0001 \
  --lora_rank 8 \
  --memory_dim 256 \
  --use_snma True \
  --use_chrc True
```
**Key H-Mem Arguments:**
- `--use_snma`: Enable Short-term Neural Memory Adapter.
- `--use_chrc`: Enable Cross-Horizon Retrieval Corrector.
- `--pogt_ratio`: Ratio of Partially-Observed Ground Truth used.
- `--memory_capacity`: Size of the memory buffer.

## 5. Key Concepts & Terminology

- **Online Learning:** The model updates its weights incrementally as new data arrives, rather than training once on a fixed dataset.
- **Concept Drift:** The statistical properties of the target variable change over time, requiring model adaptation.
- **Delayed Feedback:** In TSF, the true value for time `t+H` is only known at time `t+H`. Standard online learning assumes immediate feedback.
- **H-Mem Components:**
    - **SNMA (Short-term Neural Memory Adapter):** Adapts to immediate covariate shifts using recent partial data.
    - **CHRC (Cross-Horizon Retrieval Corrector):** Uses historical residuals to correct the final prediction, handling seasonal/recurring drift.
