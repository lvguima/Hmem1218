# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OnlineTSF** is an online time series forecasting framework that emphasizes continual model adaptation without information leakage. It implements the PROCEED method (Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting) published at KDD 2025, along with support for other online learning methods (SOLID, OneNet, FSNet, DER++, ER, and naive online gradient descent).

**Key Innovation**: The framework ensures no information leakage by updating models only with observed ground truth, not with forecasted samples. It also utilizes validation data in the online learning phase for better adaptation.

## Common Development Commands

### Running Experiments

```bash
# Basic training on a dataset
python run.py --model PatchTST --dataset ETTh1 --seq_len 96 --pred_len 24

# Online learning with naive gradient descent
python run.py --model iTransformer --dataset ETTh2 --online_method Online \
  --seq_len 336 --pred_len 24 --online_learning_rate 0.00003

# PROCEED method (main contribution)
python run.py --model iTransformer --dataset ETTh2 --online_method Proceed \
  --concept_dim 200 --bottleneck_dim 32 --tune_mode down_up

# H-Mem method (in development)
python run.py --model iTransformer --dataset ETTm1 --online_method HMem \
  --only_test --pretrain --online_learning_rate 0.00001 \
  --lora_rank 8 --lora_alpha 16.0 --memory_dim 256 --bottleneck_dim 32 \
  --memory_capacity 2000 --retrieval_top_k 5 --hmem_warmup_steps 100 \
  --freeze True --use_snma True --use_chrc True

# Run multiple iterations and aggregate results
python run.py --model PatchTST --dataset ETTh1 --itr 5 --seq_len 96 --pred_len 24

# Using provided bash scripts (organized by model/method/dataset)
bash scripts/online/iTransformer/Online/ETTh2.sh
bash scripts/pretrain/iTransformer/ETTh1.sh
```

### Key Arguments

**Essential**:
- `--model`: Forecasting backbone (PatchTST, iTransformer, Autoformer, Informer, Transformer, Crossformer, DLinear, NLinear, Linear, RLinear, TCN, FSNet, OneNet, MTGNN, GPT4TS, LIFT, LightMTS, MStream, Stat_models)
- `--dataset`: Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, ECL, Traffic, Weather, Exchange, Illness, METR_LA, PEMS_BAY, NYC_BIKE, NYC_TAXI, PeMSD4, PeMSD8, Wind, and custom datasets)
- `--seq_len`: Lookback window length (typical: 96, 336, 512)
- `--pred_len`: Forecast horizon (typical: 24, 48, 96, 168, 336, 720)

**Training**:
- `--learning_rate`: Learning rate for pretraining (default: 0.0001)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of pretraining epochs (default: 10)
- `--itr`: Number of iterations to run (default: 1)

**Online Learning**:
- `--online_method`: Method to use (Online, Proceed, SOLID, OneNet, FSNet, DER++, ER, HMem; default: Online)
- `--online_learning_rate`: Learning rate for online phase (typically higher than pretraining)
- `--border_type`: Data split strategy (default: 'online' = 20% train, 5% val, 75% test)

**PROCEED-Specific**:
- `--concept_dim`: Concept dimension $d_c$ (default: 200)
- `--bottleneck_dim`: Bottleneck dimension $r$ (default: 32)
- `--tune_mode`: Adaptation mode (down_up, up, ssf; default: down_up)

**H-Mem-Specific**:
- `--lora_rank`: LoRA rank dimension (default: 8)
- `--lora_alpha`: LoRA scaling factor (default: 16.0)
- `--memory_dim`: Neural memory dimension (default: 256)
- `--bottleneck_dim`: Bottleneck dimension for compression (default: 32)
- `--memory_capacity`: Error bank capacity for CHRC (default: 1000)
- `--retrieval_top_k`: Number of nearest neighbors to retrieve (default: 5)
- `--pogt_ratio`: Ratio of prediction horizon observed as POGT (default: 0.5)
- `--hmem_warmup_steps`: SNMA warmup steps before joint training (default: 100)
- `--freeze`: Freeze backbone model (default: True)
- `--use_snma`: Enable SNMA adapter (default: True)
- `--use_chrc`: Enable CHRC correction (default: True)
- `--chrc_temperature`: Temperature for softmax aggregation (default: 0.1)
- `--chrc_aggregation`: Aggregation method (softmax, weighted_mean; default: softmax)
- `--chrc_ramp_steps`: Gradual CHRC activation steps (default: 0)
- `--chrc_min_entries`: Minimum memory bank entries before CHRC activates (default: 0)

**Other**:
- `--use_gpu`: Use GPU (default: True)
- `--gpu`: GPU device ID (default: 0)
- `--seed`: Random seed for reproducibility (default: 2021)

## Architecture and Structure

### Directory Organization

```
adapter/              # Online adaptation modules (PEFT methods)
  ├── module/         # Base classes and specific adapters (down_up, ssf, up)
  │   ├── lora.py     # LoRA implementation (Linear, Conv1d)
  │   └── neural_memory.py  # SNMA components (SurpriseCalculator, NeuralMemoryState, LoRAHyperNetwork)
  ├── proceed.py      # PROCEED adapter combining all components
  └── hmem.py         # H-Mem adapter (SNMA + CHRC + LoRA injection)

data_provider/        # Data loading and preprocessing
  ├── data_loader.py  # Dataset classes (ETT, Custom, Recent, etc.)
  └── data_factory.py # Factory for creating datasets/dataloaders

exp/                  # Experiment runners (training pipelines)
  ├── exp_basic.py    # Base class with device management, model building
  ├── exp_main.py     # Standard pretraining pipeline
  ├── exp_online.py   # Online learning pipeline
  ├── exp_proceed.py  # PROCEED-specific online learning
  ├── exp_solid.py    # SOLID method implementation
  └── exp_hmem.py     # H-Mem method implementation

layers/               # Neural network components
  ├── AutoCorrelation.py, SelfAttention_Family.py  # Attention mechanisms
  ├── Autoformer_EncDec.py, Transformer_EncDec.py # Encoder/decoder pairs
  ├── PatchTST_*.py   # PatchTST-specific layers
  ├── RevIN.py        # Reversible Instance Normalization
  ├── Embed.py        # Embedding layers
  ├── cross_models/   # Crossformer components
  ├── ts2vec/         # TS2Vec and FSNet components
  └── graph.py        # Graph layers for MTGNN

models/               # 20+ forecasting model implementations
  ├── Transformer-based: Autoformer, Informer, Transformer, iTransformer, Crossformer
  ├── Patch-based: PatchTST
  ├── Linear: DLinear, NLinear, Linear, RLinear
  ├── Convolutional: TCN, FSNet
  ├── Graph: MTGNN
  ├── Ensemble: OneNet
  └── Others: GPT4TS, LIFT, LightMTS, MStream, Stat_models

util/                 # Utility functions
  ├── tools.py        # EarlyStopping, learning rate scheduling, model loading
  ├── metrics.py      # MSE, MAE, RMSE, MAPE calculations
  ├── timefeatures.py # Time encoding (hourly, daily, monthly, etc.)
  ├── buffer.py       # Experience replay buffer for continual learning
  ├── error_bank.py   # CHRC error memory bank implementation
  └── masking.py      # Attention masking utilities

scripts/              # Bash scripts for experiments
  ├── pretrain/       # Pretraining scripts
  └── online/         # Online learning scripts (organized by model/method/dataset)

tests/                # Unit tests
  ├── test_lora.py           # LoRA implementation tests
  ├── test_neural_memory.py  # SNMA component tests
  ├── test_error_bank.py     # CHRC component tests
  └── test_hmem_integration.py  # H-Mem integration tests

dataset/              # Data directory (CSV/HDF5 files stored here)
logs/                 # Training logs and results
checkpoints/          # Model checkpoints organized by dataset/model
```

### Key Files

| File | Purpose |
|------|---------|
| `run.py` | Main entry point with argument parser and experiment orchestration |
| `settings.py` | Central configuration: dataset paths, model hyperparameters, learning rates |
| `exp/exp_online.py` | Core online learning pipeline (23KB) |
| `exp/exp_main.py` | Standard pretraining pipeline (13KB) |
| `exp/exp_solid.py` | SOLID method implementation (16KB) |
| `exp/exp_hmem.py` | H-Mem method implementation (21KB) |
| `data_provider/data_loader.py` | Dataset classes for different formats (CSV, HDF5, NPZ) |
| `adapter/proceed.py` | PROCEED adapter implementation |
| `adapter/hmem.py` | H-Mem adapter implementation |
| `adapter/module/lora.py` | LoRA injection for parameter-efficient fine-tuning |
| `adapter/module/neural_memory.py` | SNMA components (neural memory state, surprise calculation, LoRA hypernet) |
| `util/error_bank.py` | CHRC error memory bank for historical error retrieval |

### Architectural Patterns

1. **Factory Pattern**: `data_factory.py` creates datasets and dataloaders
2. **Strategy Pattern**: Different online methods (Online, PROCEED, SOLID, H-Mem, etc.) implement common interface
3. **Adapter Pattern**: `adapter/` modules wrap models with adaptation capabilities
4. **Template Method**: `Exp_Basic` defines training template, subclasses override specific phases
5. **Lazy Loading**: `exp/__init__.py` uses `__getattr__` for delayed imports of experiment classes
6. **Wrapper Pattern**: `Dataset_Recent` wraps datasets for online learning scenarios
7. **HyperNetwork Pattern (H-Mem)**: SNMA uses a hypernetwork to generate LoRA parameters dynamically
8. **Memory Bank Pattern (H-Mem)**: CHRC maintains a retrieval-based error memory for historical pattern matching

### Data Flow

1. **Pretraining Phase**:
   - Load dataset via `data_factory.get_data()` → returns train/val/test dataloaders
   - Initialize model from `models/` directory
   - Train using `Exp_Main` pipeline
   - Save checkpoint

2. **Online Learning Phase**:
   - Load pretrained model checkpoint
   - Use `Exp_Online` (or method-specific subclass) pipeline
   - For each test sample:
     - Make prediction
     - Receive ground truth
     - Update model with ground truth (no information leakage)
     - Optionally update adapter parameters (PROCEED, SOLID, etc.)

3. **H-Mem Online Learning Phase** (special case):
   - Load pretrained backbone
   - Inject LoRA layers into backbone, freeze backbone if specified
   - Initialize SNMA (Short-term Neural Memory Adapter) and CHRC (Cross-Horizon Retrieval Corrector)
   - For each test sample:
     - Extract POGT (Partially Observed Ground Truth) from recent history
     - SNMA generates LoRA parameters from POGT via neural memory
     - Get adapted prediction with injected LoRA parameters
     - CHRC retrieves historical error patterns and applies correction
     - Make final prediction combining adaptation and correction
     - Receive full ground truth after prediction horizon
     - Update SNMA via backpropagation
     - Store error in CHRC memory bank for future retrieval
     - Update neural memory state based on surprise

### Important Design Decisions

1. **No Information Leakage**: Models are updated only with observed ground truth, never with forecasted samples
2. **Full Data Utilization**: Validation data is included in online learning phase (not held out)
3. **Data Split Strategy**: Default `border_type='online'` uses 20% train, 5% val, 75% test (vs traditional 7:2:1)
4. **Modular Model Design**: Models are independent; adapters wrap them for online learning
5. **Distributed Training**: Support for multi-GPU via DataParallel and DistributedDataParallel
6. **Mixed Precision**: AMP support for efficient training
7. **LoRA Injection (H-Mem)**: Parameter-efficient adaptation via low-rank matrix injection
8. **Delayed Memory Updates (H-Mem)**: Error bank updates respect prediction horizon delay (no information leakage)
9. **Gradient Clipping (H-Mem)**: Always clip gradients to prevent NaN loss from gradient explosion

## Configuration

### settings.py Structure

Contains three main sections:

1. **Dataset Configurations**: Path, target column, and feature count for each dataset
   ```python
   data_settings = {
       'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7], ...},
       # ... 20+ datasets
   }
   ```

2. **Model Hyperparameters**: Architecture-specific settings for each model
   ```python
   hyperparams = {
       'PatchTST': {'d_model': 768, 'n_heads': 4, ...},
       'iTransformer': {'d_model': 512, 'n_heads': 8, ...},
       # ... for each model
   }
   ```

3. **Learning Rates**: Pretraining and online learning rates for each model
   ```python
   pretrain_lr_dict = {'PatchTST': 0.0001, 'iTransformer': 0.0001, ...}
   pretrain_lr_online_dict = {'PatchTST': 0.00003, 'iTransformer': 0.00003, ...}
   ```

### Adding New Datasets

1. Place CSV/HDF5 file in `dataset/` directory
2. Add entry to `data_settings` in `settings.py`:
   ```python
   'MyDataset': {
       'data': 'mydata.csv',
       'T': 'target_column_name',
       'M': [num_features, num_features],  # [multivariate_count, multivariate_count]
       'S': [num_features, num_features],  # for univariate
       'MS': [num_features, num_features]  # for multivariate with single target
   }
   ```
3. Use `--dataset MyDataset` in run.py

### Adding New Models

1. Create model file in `models/` directory implementing PyTorch `nn.Module`
2. Add hyperparameters to `hyperparams` in `settings.py`
3. Add learning rates to `pretrain_lr_dict` and `pretrain_lr_online_dict`
4. Model will be automatically instantiated by `run.py`

## Supported Methods and Models

### Online Learning Methods

- **H-Mem**: Horizon-Bridging Neural Memory Network (in development) - combines Short-term Neural Memory Adapter (SNMA) with Cross-Horizon Retrieval Corrector (CHRC)
- **PROCEED**: Proactive Model Adaptation (KDD 2025) - main contribution
- **SOLID**: Sample-level Contextualized Adapter (KDD 2024)
- **OneNet**: Online Ensembling Network (NeurIPS 2024)
- **FSNet**: Learning Fast and Slow (ICLR 2023)
- **DER++**: Dark Experience Replay (NeurIPS 2020)
- **ER**: Experience Replay
- **Online**: Naive Online Gradient Descent

### Forecasting Models (20+)

**Transformer-based**: Autoformer, Informer, Transformer, iTransformer, Crossformer
**Patch-based**: PatchTST
**Linear**: DLinear, NLinear, Linear, RLinear
**Convolutional**: TCN, FSNet
**Graph-based**: MTGNN
**Ensemble**: OneNet
**Others**: GPT4TS, LIFT, LightMTS, MStream, Stat_models

### Supported Datasets (20+)

**Standard**: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Traffic, Weather, Exchange, Illness, Wind
**Spatial-Temporal**: METR_LA, PEMS_BAY, NYC_BIKE, NYC_TAXI, PeMSD4, PeMSD8
**Custom**: Any CSV/HDF5/NPZ file placed in `dataset/` directory

## Development Workflow

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_lora.py -v
python -m pytest tests/test_neural_memory.py -v
python -m pytest tests/test_error_bank.py -v
python -m pytest tests/test_hmem_integration.py -v

# Run specific test function
python -m pytest tests/test_lora.py::test_inject_lora_layers -v
```

### Understanding Experiment Flow

1. **Entry Point**: `run.py` parses arguments and orchestrates experiment
2. **Data Loading**: `data_factory.get_data()` creates dataloaders based on `border_type`
3. **Model Creation**: Model instantiated from `models/` with hyperparams from `settings.py`
4. **Experiment Selection**: Appropriate `Exp_*` class selected based on `online_method`
5. **Training/Testing**: `Exp_*` class handles pretraining and online phases
6. **Results**: Metrics computed and logged to `logs/` directory

### Modifying Online Learning Behavior

1. **Pretraining**: Modify `Exp_Main.train()` in `exp/exp_main.py`
2. **Online Phase**: Modify `Exp_Online.online_learning()` in `exp/exp_online.py`
3. **Method-Specific**: Subclass `Exp_Online` (see `Exp_Proceed`, `Exp_Solid` for examples)
4. **Adapter Logic**: Modify `adapter/module/` for parameter-efficient tuning

### Adding a New Online Method

1. Create `Exp_NewMethod` class in `exp/exp_newmethod.py` inheriting from `Exp_Online`
2. Override `online_learning()` method with custom update logic
3. Register in `exp/__init__.py` lazy loading
4. Add to argument parser in `run.py`
5. Add learning rate to `settings.py` if needed

### Understanding H-Mem Components

H-Mem consists of three main components working together:

**1. LoRA Injection**:
- `adapter/module/lora.py` provides `inject_lora_layers()` to add low-rank adaptation matrices to backbone
- LoRA layers are injected into Linear and Conv1d layers based on target module patterns
- Backbone can be frozen while LoRA parameters provide adaptation capacity
- LoRA parameters are dynamically generated by SNMA (not trained directly)

**2. SNMA (Short-term Neural Memory Adapter)**:
- Located in `adapter/module/neural_memory.py`
- **SurpriseCalculator**: Computes surprise score from POGT via self-supervised prediction task
- **NeuralMemoryState**: Maintains recurrent memory state updated based on surprise
- **LoRAHyperNetwork**: Generates LoRA parameters (A, B matrices) from memory state
- High surprise → fast memory update → rapid LoRA adaptation
- Low surprise → slow memory update → stable LoRA parameters

**3. CHRC (Cross-Horizon Retrieval Corrector)**:
- Located in `util/error_bank.py`
- **Memory Bank**: Stores historical (POGT features, prediction errors) pairs
- **Feature Extraction**: Encodes POGT into fixed-dimensional features via CNN + transformer
- **Retrieval**: Finds k-nearest neighbors in feature space using cosine similarity
- **Aggregation**: Combines retrieved errors via softmax or weighted mean
- **Correction**: Applies aggregated error as additive correction to adapted prediction

**Integration Flow in H-Mem**:
```
Input (x_enc, pogt) → Base Prediction (frozen backbone)
                   ↓
                SNMA: pogt → surprise → memory update → LoRA params
                   ↓
                Adapted Prediction (backbone + LoRA)
                   ↓
                CHRC: pogt → retrieve errors → correction
                   ↓
                Final Prediction = Adapted Prediction + Correction
                   ↓
                (After delay) Ground Truth → Update SNMA, Store Error in CHRC
```

### Debugging Tips

- Use `--seed` for reproducibility
- Check `logs/online/` for detailed training logs
- Use `--itr 1` for single run during development
- Print model architecture: `print(model)` in `exp_basic.py`
- Monitor GPU memory: `nvidia-smi` during training
- Verify data loading: Check shapes in `data_factory.py`

#### H-Mem Specific Debugging

**NaN Loss Issues**:
- H-Mem includes automatic NaN detection and recovery in `exp/exp_hmem.py`
- Gradient clipping is always applied (max_norm=1.0) to prevent explosion
- If NaN persists, try lowering `--online_learning_rate` (e.g., 1e-5 or 1e-6)
- Check memory bank capacity: increase `--memory_capacity` for longer history

**Component Ablation**:
- Test SNMA alone: `--use_snma True --use_chrc False`
- Test CHRC alone: `--use_snma False --use_chrc True`
- Test without adaptation: `--use_snma False --use_chrc False`

**Memory Issues**:
- Reduce `--lora_rank` (default: 8 → try 2 or 4)
- Reduce `--memory_dim` (default: 256 → try 64 or 128)
- Reduce `--memory_capacity` (default: 1000 → try 100 or 500)
- Reduce `--batch_size` if GPU OOM occurs

**Learning Rate Tuning**:
- H-Mem is sensitive to learning rate due to LoRA updates
- Start with very low rates: 1e-5 to 3e-6
- ER baseline often uses even lower: 3e-7
- Higher learning rates may cause instability in LoRA parameter generation

**CHRC Tuning**:
- Gradual activation: `--chrc_ramp_steps 1000 --chrc_min_entries 50`
- Softer aggregation: `--chrc_aggregation weighted_mean` or `--chrc_temperature 0.5`
- Reduce correction magnitude: `--chrc_max_correction 5.0` (default: inf)

## Dependencies and Environment

**Python**: 3.7+
**Core**: PyTorch 1.9+ (2.0+ for model compilation)
**GPU**: CUDA 11.0+ (optional, for GPU acceleration)

**Key Libraries**:
- torch, numpy, pandas, scikit-learn
- matplotlib (visualization)
- h5py (HDF5 file handling, optional)
- tqdm (progress bars)
- transformers (optional, for GPT4TS model)
- einops (tensor operations)
- pytest (testing, development only)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Important Notes

- **Information Leakage Prevention**: Always ensure models are updated with ground truth only, not predictions
- **Data Split**: Default `border_type='online'` is intentional for realistic online learning scenarios
- **Learning Rates**: Online learning rates are typically higher than pretraining rates (tuned on validation data)
- **Reproducibility**: Use fixed seeds and log all hyperparameters
- **GPU Memory**: Some models (Crossformer, MTGNN) are memory-intensive; reduce batch size if needed
- **Dataset Download**: Refer to `dataset/README.md` for download instructions
- **H-Mem Status**: H-Mem is currently in development and may undergo changes. Check recent bug fix docs (`1214some_bug_fix.md`) for known issues
- **Checkpoint Organization**: Checkpoints are automatically organized by `dataset/model/` to avoid conflicts
- **Default Arguments**: When running experiments, be aware that `--online_method` defaults to `None` (will only pretrain unless specified)
- **Progress Monitoring**: Use `show_progress=True` in online learning methods to see step-by-step metrics
