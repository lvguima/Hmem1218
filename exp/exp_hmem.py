"""
Experiment class for H-Mem training and evaluation.

Extends Exp_Online with H-Mem specific:
- Two-phase training (SNMA warmup + joint training)
- Delayed memory bank updates
- POGT extraction from streaming data
- Flexible adaptation strategies

Author: H-Mem Implementation
Date: 2025-12-13
"""

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from typing import Optional, Dict, Any, Tuple

from exp.exp_online import Exp_Online
from adapter.hmem import HMem, build_hmem
from data_provider.data_factory import get_dataset, get_dataloader
from data_provider.data_loader import Dataset_Recent


class Exp_HMem(Exp_Online):
    """
    H-Mem experiment class.

    Extends Exp_Online with H-Mem specific features:
    - Phased training (SNMA warmup → joint training)
    - Delayed memory bank updates (respecting feedback delay)
    - POGT extraction utilities
    - Flexible component control (enable/disable SNMA or CHRC)
    """

    def __init__(self, args):
        # Add H-Mem specific args if not present
        if not hasattr(args, 'freeze'):
            args.freeze = True  # Freeze backbone by default

        # H-Mem specific settings (MUST be set before super().__init__)
        self.pogt_ratio = getattr(args, 'pogt_ratio', 0.5)
        self.warmup_steps = getattr(args, 'hmem_warmup_steps', 100)
        self.joint_training = getattr(args, 'hmem_joint_training', True)
        self.use_snma = getattr(args, 'use_snma', False)
        self.use_chrc = getattr(args, 'use_chrc', True)

        self.pogt_source = getattr(args, 'hmem_pogt_source', 'batch_x')
        if self.pogt_source not in ('batch_x', 'batch_y'):
            raise ValueError("hmem_pogt_source must be 'batch_x' or 'batch_y'")

        # Delayed update buffer
        # Stores (step, pogt, prediction) tuples waiting for ground truth
        self.pending_updates = []
        self.delay_steps = args.pred_len  # Wait for full horizon

        # Training phase tracking
        self._warmup_phase = True
        self._current_step = 0

        super().__init__(args)

    def _build_model(self, model=None, framework_class=None):
        """Build model with H-Mem wrapper."""
        # Build backbone first using parent's method
        # But don't wrap with any framework yet
        backbone = super()._build_model(model, framework_class=None)

        # Now wrap with H-Mem
        hmem_model = build_hmem(backbone, self.args)

        # Apply multi-GPU if needed
        if self.args.use_multi_gpu and self.args.use_gpu:
            hmem_model = nn.DataParallel(hmem_model, device_ids=self.args.device_ids)

        return hmem_model.to(self.device)

    def _select_optimizer(self, filter_frozen: bool = True, return_self: bool = True, model=None):
        """
        Create optimizer with different learning rates for components.

        H-Mem has three trainable components:
        1. SNMA (neural memory)
        2. CHRC (retrieval corrector)
        3. Backbone (usually frozen)

        Args:
            filter_frozen: Whether to filter frozen parameters (ignored for H-Mem)
            return_self: Whether to return cached optimizer (ignored for H-Mem)
            model: Model to optimize. If provided during init, we skip optimizer creation.

        Note:
            During _build_model(), parent class may call this with a backbone model.
            We return None because H-Mem wraps the backbone and creates optimizer later.
        """
        # If called during _build_model() with a model parameter (backbone)
        # We can't create the optimizer yet because HMem wrapper doesn't exist
        if model is not None and not hasattr(self, 'model'):
            # During initialization, just return None
            # Optimizer will be created in online() method
            return None

        # After initialization, create optimizer for HMem components
        if model is None:
            if not hasattr(self, 'model'):
                return None
            model = self._model.module if hasattr(self._model, 'module') else self._model
        elif hasattr(model, 'module'):
            model = model.module

        # Collect parameters
        param_groups = []

        # SNMA parameters
        if self.use_snma:
            snma_params = list(model.snma.parameters())
            if snma_params:
                param_groups.append({
                    'params': snma_params,
                    'lr': self.args.online_learning_rate,
                    'name': 'snma'
                })

        # CHRC parameters
        if self.use_chrc and model.chrc is not None:
            chrc_params = list(model.chrc.parameters())
            if chrc_params:
                # Use lower learning rate for CHRC
                param_groups.append({
                    'params': chrc_params,
                    'lr': self.args.online_learning_rate * 0.5,
                    'name': 'chrc'
                })

        # Backbone parameters (if not frozen)
        if not model.freeze_backbone:
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
                    'lr': self.args.online_learning_rate * 0.1,
                    'name': 'backbone'
                })

        if not param_groups:
            # Fallback: collect all trainable params
            all_params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{'params': all_params, 'lr': self.args.online_learning_rate}]

        return optim.AdamW(
            param_groups,
            weight_decay=getattr(self.args, 'weight_decay', 0.01)
        )

    def _extract_pogt(
        self,
        batch_seq: torch.Tensor,
        full_gt: bool = False
    ) -> torch.Tensor:
        """
        Extract POGT from batch data.

        Args:
            batch_seq: Source sequence [batch, total_len, features]
            full_gt: Whether batch_seq contains full ground truth

        Returns:
            POGT tensor [batch, pogt_len, features]
        """
        model = self._model.module if hasattr(self._model, 'module') else self._model
        pogt_len = model.pogt_len

        if full_gt:
            # Take the first pogt_len steps from full GT
            if batch_seq.size(1) >= pogt_len:
                return batch_seq[:, :pogt_len, :]
            else:
                # Pad if needed
                padding = torch.zeros(
                    batch_seq.size(0),
                    pogt_len - batch_seq.size(1),
                    batch_seq.size(2),
                    device=batch_seq.device
                )
                return torch.cat([batch_seq, padding], dim=1)
        else:
            # Take the last pogt_len steps (most recent observations)
            if batch_seq.size(1) >= pogt_len:
                return batch_seq[:, -pogt_len:, :]
            else:
                # Pad at beginning
                padding = torch.zeros(
                    batch_seq.size(0),
                    pogt_len - batch_seq.size(1),
                    batch_seq.size(2),
                    device=batch_seq.device
                )
                return torch.cat([padding, batch_seq], dim=1)

    def _update_online(
        self,
        batch: tuple,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
        **kwargs
    ) -> float:
        """
        Online update step for H-Mem.

        Args:
            batch: (batch_x, batch_y, batch_x_mark, batch_y_mark)
            criterion: Loss function
            optimizer: Optimizer
            scaler: Optional AMP scaler

        Returns:
            Loss value
        """
        self._model.train()
        model = self._model.module if hasattr(self._model, 'module') else self._model

        # Unpack batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if batch_x_mark is not None:
            batch_x_mark = batch_x_mark.float().to(self.device)

        # Extract POGT from observations
        # In online setting, we use recent observations as POGT
        pogt_source = batch_x if self.pogt_source == 'batch_x' else batch_y
        pogt = self._extract_pogt(pogt_source, full_gt=False)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Get model output with components
            outputs = model(
                batch_x, batch_x_mark,
                pogt=pogt,
                return_components=True
            )

            prediction = outputs['prediction']

            # Guard: NaN in prediction
            if torch.isnan(prediction).any():
                print("[Warning] NaN detected in prediction, skipping update")
                if hasattr(model, 'snma'):
                    model.snma.reset(batch_size=batch_x.size(0))
                return 0.0

            # Main loss: prediction vs ground truth
            # Ground truth is the last pred_len steps
            gt = batch_y[:, -self.args.pred_len:, :]
            loss = criterion(prediction, gt)

            # Guard: NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("[Warning] NaN/Inf detected in loss, skipping update")
                if hasattr(model, 'snma'):
                    model.snma.reset(batch_size=batch_x.size(0))
                return 0.0

            # Auxiliary loss: encourage adaptation to improve base prediction
            if self.use_snma and 'adapted_prediction' in outputs:
                base_loss = criterion(outputs['base_prediction'], gt)
                adapted_loss = criterion(outputs['adapted_prediction'], gt)

                adaptation_gain = base_loss - adapted_loss
                if adaptation_gain < 0:
                    loss = loss + 0.1 * torch.relu(-adaptation_gain)

        # Backward pass - only if loss requires grad
        if loss.requires_grad:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if hasattr(self.args, 'max_grad_norm'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # IMPORTANT: Detach SNMA memory to free computation graph without wiping state
        if hasattr(model, 'snma'):
            model.snma.detach_state()
        elif hasattr(model, 'module') and hasattr(model.module, 'snma'):
            model.module.snma.detach_state()

        # Store for delayed memory bank update
        if model.flag_store_errors:
            bucket_id = None
            if isinstance(model.chrc, nn.ModuleList) and hasattr(model, '_compute_bucket_id'):
                bucket_id = model._compute_bucket_id(batch_x_mark)
            self.pending_updates.append({
                'step': self._current_step,
                'pogt': pogt.detach(),
                'prediction': prediction.detach(),
                'bucket_id': bucket_id,
            })

        return loss.item()

    def _process_delayed_updates(
        self,
        current_step: int,
        ground_truth: torch.Tensor
    ):
        """
        Process pending memory bank updates when GT becomes available.

        Args:
            current_step: Current online step
            ground_truth: Full ground truth that just became available
        """
        model = self._model.module if hasattr(self._model, 'module') else self._model

        if not self.use_chrc or model.chrc is None:
            return

        # Find updates that are ready (delay has passed)
        ready_updates = [
            u for u in self.pending_updates
            if current_step - u['step'] >= self.delay_steps
        ]

        for update in ready_updates:
            # Use the stored POGT and prediction from the update
            stored_pogt = update['pogt']
            stored_prediction = update['prediction']

            # Compute error using the stored prediction
            error = ground_truth.to(stored_prediction.device) - stored_prediction

            # Store the error with the correct POGT in memory bank
            chrc_module = model.chrc
            if isinstance(model.chrc, nn.ModuleList) and hasattr(model, '_get_chrc_for_bucket'):
                chrc_module = model._get_chrc_for_bucket(update.get('bucket_id'))
            if chrc_module is None:
                self.pending_updates.remove(update)
                continue

            chrc_module.store_error(
                stored_pogt,
                error
            )

            # Update cold start flag
            if model._is_cold_start:
                model._is_cold_start = torch.tensor(False)

            self.pending_updates.remove(update)

    def _get_chrc_mem_size(self, model: nn.Module) -> int:
        if not self.use_chrc or model.chrc is None:
            return 0
        if isinstance(model.chrc, nn.ModuleList):
            return sum(chrc.memory_bank.current_size for chrc in model.chrc)
        return model.chrc.memory_bank.current_size

    def update_valid(self, valid_data=None):
        """
        Adapt on validation set before testing.

        Uses validation data to:
        1. Warm up the neural memory
        2. Populate the error memory bank
        """
        self.phase = 'online'

        if valid_data is None or not isinstance(valid_data, Dataset_Recent):
            valid_data = get_dataset(
                self.args, 'val', self.device,
                wrap_class=self.args.wrap_data_class + [Dataset_Recent],
                take_post=self.args.pred_len - 1,
                **self.wrap_data_kwargs
            )

        valid_loader = get_dataloader(valid_data, self.args, 'online')
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler() if self.args.use_amp else None

        self.model.train()
        print(f"[H-Mem] Warming up on validation data ({len(valid_loader)} batches)...")

        warmup_pbar = tqdm(
            valid_loader,
            desc='H-Mem Validation Warmup',
            mininterval=10,
            leave=False,
            dynamic_ncols=True
        )
        warmup_losses = []

        for i, (recent_batch, current_batch) in enumerate(warmup_pbar):
            # Update with recent batch
            loss = self._update_online(recent_batch, criterion, model_optim, scaler)
            warmup_losses.append(loss)

            # Process delayed updates if GT available
            if len(self.pending_updates) > 0:
                # Use current_batch ground truth for delayed update
                _, batch_y, _, _ = current_batch
                self._process_delayed_updates(i, batch_y[:, -self.args.pred_len:, :])

            if (i + 1) % 100 == 0: # Update postfix every 100 steps
                avg_loss = np.mean(warmup_losses[-100:]) if warmup_losses else 0
                mem_size = self._get_chrc_mem_size(self._model.module if hasattr(self._model, 'module') else self._model)
                warmup_pbar.set_postfix(
                    step=i + 1,
                    loss=f"{avg_loss:.3f}",
                    mem_size=mem_size
                )

        if self.args.local_rank <= 0:
            mem_size = self._get_chrc_mem_size(self._model.module if hasattr(self._model, 'module') else self._model)
            print(f"[H-Mem] Validation warmup complete. Memory bank size: {mem_size}")


    def online(
        self,
        online_data=None,
        target_variate: int = 0,
        phase: str = 'test',
        show_progress: bool = False
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Main online learning loop for H-Mem.

        Args:
            online_data: Online dataset
            target_variate: Target variable index
            phase: 'val', 'test', or 'online'
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (MSE, predictions, ground_truths)
        """
        model = self._model.module if hasattr(self._model, 'module') else self._model
        criterion = self._select_criterion()
        optimizer = self._select_optimizer()

        # Optional: learning rate scheduler
        if hasattr(self.args, 'use_scheduler') and self.args.use_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=len(online_data) // self.args.batch_size,
                eta_min=self.args.online_learning_rate * 0.01
            )
        else:
            scheduler = None

        # Reset state for new phase
        if phase in ['test', 'online']:
            model.reset_memory(batch_size=self.args.batch_size)
            self.pending_updates = []
            self._current_step = 0
            self._warmup_phase = True

        # Tracking
        preds = []
        trues = []
        losses = []

        # Progress print interval
        print_interval = 500  # Print every 500 steps

        # Create dataloader
        if online_data is None:
            online_data, online_loader = self._get_data(phase)
        else:
            online_loader = get_dataloader(online_data, self.args, 'online')

        # Create progress bar
        pbar = tqdm(
            online_loader,
            desc=f'H-Mem Online ({phase})',
            disable=not show_progress,
            mininterval=10,
            leave=False,          # Clear progress bar after loop
            dynamic_ncols=True,   # Adjust columns dynamically
        )

        # Training loop
        for i, (recent_batch, current_batch) in enumerate(pbar):
            self._current_step = i
            pogt_for_pred = None

            # Phase switching: warmup → joint training
            if self._warmup_phase and i >= self.warmup_steps:
                self._warmup_phase = False
                if self.joint_training and self.use_chrc and self.use_snma:
                    model.freeze_chrc(False)
                    if self.args.local_rank <= 0:
                        pbar.write(f"[H-Mem] Warmup complete at step {i}. "
                                   f"Enabling joint SNMA + CHRC training.")
                elif self.use_chrc and not self.use_snma:
                    model.freeze_chrc(False)
                    if self.args.local_rank <= 0:
                        pbar.write(f"[H-Mem] Warmup complete at step {i}. "
                                   f"Using CHRC only (SNMA disabled).")
                elif self.use_snma and not self.use_chrc:
                    if self.args.local_rank <= 0:
                        pbar.write(f"[H-Mem] Warmup complete at step {i}. "
                                   f"Using SNMA only (CHRC disabled).")
                else:
                    if self.args.local_rank <= 0:
                        pbar.write(f"[H-Mem] Warmup complete at step {i}. "
                                   f"Both SNMA and CHRC disabled.")

            # During warmup, only train SNMA
            if self._warmup_phase and self.use_chrc:
                model.freeze_chrc(True)

            # Update model with recent observed data
            if recent_batch is not None:
                loss = self._update_online(recent_batch, criterion, optimizer, scaler=None)
                losses.append(loss)

                # Process delayed memory bank updates
                recent_x, recent_y, _, _ = recent_batch
                if len(self.pending_updates) > 0:
                    self._process_delayed_updates(i, recent_y[:, -self.args.pred_len:, :])

                # Build POGT for prediction from selected source (default: batch_x)
                recent_x = recent_x.float().to(self.device)
                recent_y = recent_y.float().to(self.device)
                pogt_source = recent_x if self.pogt_source == 'batch_x' else recent_y
                pogt_for_pred = self._extract_pogt(pogt_source, full_gt=False)

            # Make prediction for current window
            with torch.no_grad():
                model.eval()
                batch_x, batch_y, batch_x_mark, batch_y_mark = current_batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if batch_x_mark is not None:
                    batch_x_mark = batch_x_mark.float().to(self.device)

                # Use POGT from observed recent batch only
                pogt = pogt_for_pred

                # Predict
                pred = model(batch_x, batch_x_mark, pogt=pogt, return_components=False)

                model.train()

            # Collect predictions and ground truths
            preds.append(pred.detach().cpu().numpy())
            trues.append(batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy())

            # Update progress bar postfix every print_interval steps
            if (i + 1) % print_interval == 0 and self.args.local_rank <= 0:
                avg_loss = np.mean(losses[-print_interval:]) if losses else 0
                recent_preds = np.concatenate(preds[-print_interval:], axis=0)
                recent_trues = np.concatenate(trues[-print_interval:], axis=0)
                running_mse = np.mean((recent_preds - recent_trues) ** 2)
                pbar.set_postfix(
                    step=i + 1,
                    loss=f"{avg_loss:.3f}",
                    mse=f"{running_mse:.3f}"
                )

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

        # Aggregate results
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Compute metrics
        from util.metrics import metric
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        true_mean = np.mean(trues)
        true_var = np.mean((trues - true_mean) ** 2)
        if true_var > 1e-10:
            r2 = 1 - (mse / true_var)
        else:
            r2 = 0.0

        if self.args.local_rank <= 0:
            print(f"\n[H-Mem] {phase.upper()} Results:")
            print(
                "  MSE: {:.6f} | MAE: {:.6f} | RMSE: {:.6f} | RSE: {:.6f} | "
                "R2: {:.6f} | MAPE: {:.6f}".format(
                    mse, mae, rmse, rse, r2, mape
                )
            )
            if self.use_chrc and hasattr(model, 'chrc') and model.chrc is not None:
                stats = model.get_statistics()
                memory_stats = stats.get('memory_bank')
                if isinstance(memory_stats, list):
                    total_entries = sum(s.get('num_entries', 0) for s in memory_stats)
                    total_capacity = sum(s.get('capacity', 0) for s in memory_stats)
                    utilization = (total_entries / total_capacity) if total_capacity > 0 else 0.0
                else:
                    total_entries = memory_stats.get('num_entries', 0)
                    total_capacity = memory_stats.get('capacity', 0)
                    utilization = memory_stats.get('utilization', 0.0)
                print(f"  Memory Bank: {total_entries} entries, {utilization*100:.1f}% full")

        # Return format consistent with Exp_Online.online()
        return mse, mae, online_data


    def test(self, setting, test=0):
        """Override test to use online learning."""
        # Get test data
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('[H-Mem] Loading checkpoint...')
            checkpoint_path = self._checkpoint_path()
            self.model.load_state_dict(torch.load(checkpoint_path))

        # Update on validation data first
        if not hasattr(self.args, 'skip_valid_update') or not self.args.skip_valid_update:
            self.update_valid()

        # Run online test
        mse, preds, trues = self.online(
            online_data=test_data,
            phase='test',
            show_progress=True
        )

        # Save results
        if self.args.local_rank <= 0:
            folder_path = f'./results/{setting}/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path + 'metrics.npy', np.array([mse]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return mse
