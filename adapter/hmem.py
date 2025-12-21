"""
H-Mem: Horizon-Bridging Neural Memory Network for Online Time Series Forecasting.

Main module integrating:
- Frozen backbone with LoRA injection
- SNMA (Short-term Neural Memory Adapter)
- CHRC (Cross-Horizon Retrieval Corrector)
- Fusion mechanism for combining adaptations

Author: H-Mem Implementation
Date: 2025-12-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

from adapter.module.lora import (
    inject_lora_layers,
    LoRALinear,
    LoRAConv1d,
    get_lora_param_dims,
    set_all_lora_params,
    clear_all_lora_params,
    collect_lora_layers
)
from adapter.module.neural_memory import SNMA
from util.error_bank import CHRC


class HMem(nn.Module):
    """
    H-Mem: Horizon-Bridging Neural Memory Network.

    Architecture:
    1. Frozen Backbone with LoRA injection points
    2. SNMA: Generates LoRA params from POGT via neural memory
    3. CHRC: Retrieves historical error patterns for correction
    4. Fusion mechanism: Intelligently combines adapted prediction with correction

    Args:
        backbone: Pre-trained forecasting model (PatchTST, iTransformer, etc.)
        args: Configuration arguments
    """

    def __init__(self, backbone: nn.Module, args):
        super().__init__()

        self.args = args

        # Basic configurations
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.enc_in = args.enc_in  # Number of features

        # H-Mem specific parameters
        self.lora_rank = getattr(args, 'lora_rank', 8)
        self.lora_alpha = getattr(args, 'lora_alpha', 16.0)
        self.lora_ema_decay = getattr(args, 'lora_ema_decay', 0.0)
        self.memory_dim = getattr(args, 'memory_dim', 256)
        self.bottleneck_dim = getattr(args, 'bottleneck_dim', 32)
        self.memory_capacity = getattr(args, 'memory_capacity', 1000)
        self.retrieval_top_k = getattr(args, 'retrieval_top_k', 5)
        self.pogt_ratio = getattr(args, 'pogt_ratio', 0.5)
        self.chrc_feature_dim = getattr(args, 'chrc_feature_dim', 128)
        self.share_pogt = getattr(args, 'hmem_share_pogt', False)
        self.use_chrc = getattr(args, 'use_chrc', True)
        self.freeze_backbone = getattr(args, 'freeze', True)
        self.chrc_use_buckets = getattr(args, 'chrc_use_buckets', False)
        self.chrc_bucket_num = max(1, int(getattr(args, 'chrc_bucket_num', 4)))
        self.timeenc = getattr(args, 'timeenc', 1)
        self.freq = getattr(args, 'freq', 'h')

        # Calculate POGT length
        self.pogt_len = max(1, int(self.pred_len * self.pogt_ratio))
        # If sharing POGT representations, align CHRC feature dim with SNMA encoding dim
        if self.share_pogt and self.chrc_feature_dim != self.memory_dim:
            self.chrc_feature_dim = self.memory_dim
            args.chrc_feature_dim = self.chrc_feature_dim

        # 1. Inject LoRA layers into backbone and freeze if needed
        if self.freeze_backbone:
            backbone.requires_grad_(False)

        self.backbone, self.lora_layer_info = inject_lora_layers(
            backbone,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=getattr(args, 'lora_dropout', 0.0),
            target_modules=getattr(args, 'lora_target_modules', None),
            freeze_weight=self.freeze_backbone
        )

        # Get LoRA parameter dimensions
        lora_dims = self._collect_lora_dims()
        total_lora_params = sum(d['total'] for d in lora_dims.values())

        # 2. SNMA: Short-term Neural Memory Adapter (Original HyperNetwork-based)
        self.snma = SNMA(
            input_features=self.enc_in,
            memory_dim=self.memory_dim,
            bottleneck_dim=self.bottleneck_dim,
            momentum=getattr(args, 'memory_momentum', 0.9),
            num_heads=getattr(args, 'memory_num_heads', 4)
        )
        self.snma.register_lora_layers(lora_dims)

        # 3. CHRC: Cross-Horizon Retrieval Corrector
        if self.use_chrc:
            def build_chrc():
                return CHRC(
                    num_features=self.enc_in,
                    horizon=self.pred_len,
                    pogt_len=self.pogt_len,
                    feature_dim=self.chrc_feature_dim,
                    capacity=self.memory_capacity,
                    top_k=self.retrieval_top_k,
                    temperature=getattr(args, 'chrc_temperature', 0.1),
                    aggregation=getattr(args, 'chrc_aggregation', 'softmax'),
                    use_refinement=getattr(args, 'chrc_use_refinement', True),
                    trust_threshold=getattr(args, 'chrc_trust_threshold', 0.5),
                    gate_steepness=getattr(args, 'chrc_gate_steepness', 10.0),
                    use_horizon_mask=getattr(args, 'chrc_use_horizon_mask', False),
                    horizon_mask_mode=getattr(args, 'chrc_horizon_mask_mode', 'exp'),
                    horizon_mask_decay=getattr(args, 'chrc_horizon_mask_decay', 0.98),
                    horizon_mask_min=getattr(args, 'chrc_horizon_mask_min', 0.2),
                    min_similarity=getattr(args, 'chrc_min_similarity', 0.0),
                    forget_decay=getattr(args, 'chrc_forget_decay', 1.0),
                    forget_threshold=getattr(args, 'chrc_forget_threshold', 0.0),
                    max_age=getattr(args, 'chrc_max_age', 0)
                )

            if self.chrc_use_buckets and self.chrc_bucket_num > 1:
                self.chrc = nn.ModuleList([build_chrc() for _ in range(self.chrc_bucket_num)])
            else:
                self.chrc = build_chrc()
        else:
            self.chrc = None

        # State tracking
        self.register_buffer('_is_cold_start', torch.tensor(True))
        self._last_pogt: Optional[torch.Tensor] = None
        self._last_prediction: Optional[torch.Tensor] = None
        self._last_bucket_id: Optional[int] = None
        self._mode = 'train'  # 'train', 'eval', 'online'
        self._lora_ema_params: Dict[str, torch.Tensor] = {}

        # Flags for controlling behavior (read from args!)
        self.flag_use_snma = getattr(args, 'use_snma', False)
        self.flag_use_chrc = self.use_chrc
        self.flag_store_errors = True

    def _collect_lora_dims(self) -> Dict[str, Dict]:
        """Collect LoRA parameter dimensions from injected layers."""
        dims = {}
        for name, layer_info in self.lora_layer_info.items():
            dims[name] = {
                'A': layer_info['shapes']['A'],
                'B': layer_info['shapes']['B'],
                'total': layer_info['param_count'],
                'type': layer_info['type']
            }
        return dims

    def _inject_lora_params(self, lora_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """Inject generated LoRA parameters into backbone layers."""
        set_all_lora_params(self.backbone, lora_params)

    def _clear_lora_params(self):
        """Clear LoRA parameters from backbone layers."""
        clear_all_lora_params(self.backbone)

    def _ema_update(self, key: str, current: torch.Tensor) -> torch.Tensor:
        """EMA update for a LoRA tensor while keeping gradients for current step."""
        decay = self.lora_ema_decay
        prev = self._lora_ema_params.get(key)
        if prev is None or prev.shape != current.shape or prev.device != current.device:
            prev = current.detach()
        ema = decay * prev + (1 - decay) * current
        self._lora_ema_params[key] = ema.detach()
        return ema

    def _smooth_lora_params(
        self,
        lora_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Apply EMA smoothing to LoRA parameters if enabled."""
        if self.lora_ema_decay <= 0:
            return lora_params
        smoothed = {}
        for name, (A, B) in lora_params.items():
            smoothed[name] = (
                self._ema_update(f'{name}.A', A),
                self._ema_update(f'{name}.B', B),
            )
        return smoothed

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        x_dec: Optional[torch.Tensor] = None,
        x_mark_dec: Optional[torch.Tensor] = None,
        pogt: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional POGT for adaptation.

        Args:
            x_enc: Input sequence [batch, seq_len, features]
            x_mark_enc: Time features [batch, seq_len, time_features] (optional)
            x_dec: Decoder input (for decoder models, optional)
            x_mark_dec: Decoder time features (optional)
            pogt: Partially observed ground truth [batch, pogt_len, features] (optional)
            return_components: Whether to return intermediate components

        Returns:
            If return_components=False:
                prediction: [batch, pred_len, features]
            If return_components=True:
                Dictionary with all intermediate outputs
        """
        batch_size = x_enc.size(0)
        outputs = {}

        # Step 1: Get base prediction (no LoRA)
        self._clear_lora_params()
        with torch.set_grad_enabled(self.training):
            if x_mark_enc is not None:
                base_pred = self.backbone(x_enc, x_mark_enc)
            else:
                base_pred = self.backbone(x_enc)

        outputs['base_prediction'] = base_pred

        # If no POGT or not using adaptations, return base prediction
        if pogt is None or (not self.flag_use_snma and not self.flag_use_chrc):
            outputs['prediction'] = base_pred
            if return_components:
                outputs['adapted_prediction'] = base_pred
                outputs['correction'] = torch.zeros_like(base_pred)
                outputs['final_prediction'] = base_pred
            return outputs if return_components else base_pred

        # Step 2: SNMA - Generate LoRA params from POGT (Original HyperNetwork-based)
        pogt_features = None
        if self.flag_use_snma:
            if self.share_pogt:
                lora_params, diagnostics = self.snma(pogt, return_diagnostics=True)
                memory_state = diagnostics['memory_state']
                pogt_features = diagnostics['encoding']
            else:
                lora_params, memory_state = self.snma(pogt)
                pogt_features = None
            lora_params = self._smooth_lora_params(lora_params)
            outputs['memory_state'] = memory_state

            # Step 3: Get adapted prediction (with LoRA)
            self._inject_lora_params(lora_params)
            with torch.set_grad_enabled(self.training):
                if x_mark_enc is not None:
                    adapted_pred = self.backbone(x_enc, x_mark_enc)
                else:
                    adapted_pred = self.backbone(x_enc)
            self._clear_lora_params()
        else:
            adapted_pred = base_pred
            outputs['memory_state'] = None

        outputs['adapted_prediction'] = adapted_pred

        # Step 4: CHRC - Retrieve and apply historical error correction
        if self.flag_use_chrc and self.chrc is not None:
            chrc_module = self._select_chrc_module(x_mark_enc)
            if chrc_module is None or self._is_cold_start or chrc_module.memory_bank.is_empty:
                # Cold start: no historical data
                corrected_pred = adapted_pred
                outputs['correction'] = torch.zeros_like(adapted_pred)
                outputs['chrc_confidence'] = torch.zeros(batch_size, 1, device=x_enc.device)
            else:
                # Apply CHRC correction
                corrected_pred, chrc_details = chrc_module(
                    adapted_pred,
                    pogt,
                    pogt_features=pogt_features,
                    return_details=True
                )

                outputs['correction'] = chrc_details['correction']
                outputs['chrc_confidence'] = chrc_details['effective_confidence']
                outputs['chrc_details'] = chrc_details
        else:
            corrected_pred = adapted_pred
            outputs['correction'] = torch.zeros_like(adapted_pred)

        outputs['prediction'] = corrected_pred
        outputs['final_prediction'] = corrected_pred

        # Store for delayed memory bank update
        # Store corrected prediction for delayed error logging
        if self.flag_store_errors and pogt is not None:
            self._last_pogt = pogt.detach()
            self._last_prediction = corrected_pred.detach()
            self._last_bucket_id = self._compute_bucket_id(x_mark_enc)

        if return_components:
            return outputs
        else:
            return corrected_pred

    def update_memory_bank(self, ground_truth: torch.Tensor):
        """
        Update error memory bank when full ground truth becomes available.

        Called H steps after prediction, when GT is revealed.

        Args:
            ground_truth: Full horizon ground truth [batch, pred_len, features]
        """
        if not self.flag_use_chrc or self.chrc is None:
            return

        if self._last_pogt is None or self._last_prediction is None:
            return

        # Compute prediction error
        error = ground_truth - self._last_prediction

        chrc_module = self._get_chrc_for_bucket(self._last_bucket_id)
        if chrc_module is None:
            return

        # Store in CHRC memory bank
        chrc_module.store_error(
            self._last_pogt,
            error
        )

        # Update cold start flag
        if self._is_cold_start:
            self._is_cold_start = torch.tensor(False)

        # Clear cache
        self._last_pogt = None
        self._last_prediction = None
        self._last_bucket_id = None

    def reset_memory(self, batch_size: int = 1):
        """Reset all memory states (for new sequence/dataset)."""
        self.snma.reset(batch_size=batch_size)
        if isinstance(self.chrc, nn.ModuleList):
            for chrc in self.chrc:
                chrc.reset()
        elif self.chrc is not None:
            self.chrc.reset()
        self._is_cold_start = torch.tensor(True)
        self._last_pogt = None
        self._last_prediction = None
        self._last_bucket_id = None
        self._lora_ema_params = {}

    def freeze_snma(self, freeze: bool = True):
        """Freeze/unfreeze SNMA parameters."""
        for param in self.snma.parameters():
            param.requires_grad = not freeze

    def freeze_chrc(self, freeze: bool = True):
        """Freeze/unfreeze CHRC parameters."""
        if isinstance(self.chrc, nn.ModuleList):
            for chrc in self.chrc:
                for param in chrc.parameters():
                    param.requires_grad = not freeze
        elif self.chrc is not None:
            for param in self.chrc.parameters():
                param.requires_grad = not freeze

    def freeze_all_adapters(self, freeze: bool = True):
        """Freeze/unfreeze all adapter parameters (SNMA + CHRC)."""
        self.freeze_snma(freeze)
        self.freeze_chrc(freeze)

    def enable_snma(self, enable: bool = True):
        """Enable/disable SNMA adaptation."""
        self.flag_use_snma = enable

    def enable_chrc(self, enable: bool = True):
        """Enable/disable CHRC correction."""
        self.flag_use_chrc = enable and self.use_chrc

    def _compute_bucket_id(self, x_mark_enc: Optional[torch.Tensor]) -> int:
        """Compute bucket id from time features."""
        if not self.chrc_use_buckets or x_mark_enc is None:
            return 0

        time_feat = x_mark_enc[:, -1, :]
        hour = None
        day_of_week = None
        season = None

        if self.timeenc == 0:
            freq = str(self.freq).lower()
            if time_feat.size(-1) >= 1:
                season = time_feat[:, 0] - 1
            if freq in {'h', 't'} and time_feat.size(-1) >= 4:
                hour = time_feat[:, 3]
            if freq in {'h', 't', 'd', 'b'} and time_feat.size(-1) >= 3:
                day_of_week = time_feat[:, 2]
        elif self.timeenc == 1:
            raw_freq = str(self.freq).upper()
            if raw_freq.endswith('MIN'):
                freq = 'T'
            elif raw_freq.endswith('H'):
                freq = 'H'
            elif raw_freq.endswith('D'):
                freq = 'D'
            elif raw_freq.endswith('B'):
                freq = 'B'
            elif raw_freq.endswith('W'):
                freq = 'W'
            elif raw_freq.endswith('M'):
                freq = 'M'
            elif raw_freq.endswith('S'):
                freq = 'S'
            else:
                freq = raw_freq
            if freq == 'H' and time_feat.size(-1) >= 1:
                hour = (time_feat[:, 0] + 0.5) * 23.0
            elif freq == 'T' and time_feat.size(-1) >= 2:
                hour = (time_feat[:, 1] + 0.5) * 23.0
            elif freq == 'S' and time_feat.size(-1) >= 3:
                hour = (time_feat[:, 2] + 0.5) * 23.0

            if freq == 'H' and time_feat.size(-1) >= 2:
                day_of_week = (time_feat[:, 1] + 0.5) * 6.0
            elif freq == 'T' and time_feat.size(-1) >= 3:
                day_of_week = (time_feat[:, 2] + 0.5) * 6.0
            elif freq == 'S' and time_feat.size(-1) >= 4:
                day_of_week = (time_feat[:, 3] + 0.5) * 6.0
            elif freq in {'D', 'B'} and time_feat.size(-1) >= 1:
                day_of_week = (time_feat[:, 0] + 0.5) * 6.0

            if freq in {'M', 'Q'} and time_feat.size(-1) >= 1:
                season = torch.floor((time_feat[:, 0] + 0.5) * 11.0)
            elif freq == 'H' and time_feat.size(-1) >= 4:
                day_of_year = (time_feat[:, 3] + 0.5) * 364.0
                season = torch.floor(day_of_year / 365.0 * 12.0)
            elif freq == 'T' and time_feat.size(-1) >= 5:
                day_of_year = (time_feat[:, 4] + 0.5) * 364.0
                season = torch.floor(day_of_year / 365.0 * 12.0)
            elif freq == 'S' and time_feat.size(-1) >= 6:
                day_of_year = (time_feat[:, 5] + 0.5) * 364.0
                season = torch.floor(day_of_year / 365.0 * 12.0)
            elif freq in {'D', 'B'} and time_feat.size(-1) >= 3:
                day_of_year = (time_feat[:, 2] + 0.5) * 364.0
                season = torch.floor(day_of_year / 365.0 * 12.0)
            elif freq == 'W' and time_feat.size(-1) >= 2:
                week_of_year = (time_feat[:, 1] + 0.5) * 52.0
                season = torch.floor(week_of_year / 53.0 * 12.0)
        else:
            if time_feat.size(-1) >= 6:
                hour = time_feat[:, 1]
                day_of_week = time_feat[:, 2]
                season = time_feat[:, 5] - 1

        def _quantize(value: Optional[torch.Tensor], max_value: int) -> Optional[torch.Tensor]:
            if value is None:
                return None
            value = torch.floor(value).long()
            return torch.clamp(value, min=0, max=max_value)

        hour = _quantize(hour, 23)
        day_of_week = _quantize(day_of_week, 6)
        season = _quantize(season, 11)

        if hour is None and day_of_week is None and season is None:
            return 0

        bucket_key = torch.zeros(time_feat.size(0), device=time_feat.device, dtype=torch.long)
        if season is not None:
            bucket_key = bucket_key * 12 + season
        if day_of_week is not None:
            bucket_key = bucket_key * 7 + day_of_week
        if hour is not None:
            bucket_key = bucket_key * 24 + hour

        bucket = bucket_key % self.chrc_bucket_num
        return int(bucket[0].item())

    def _get_chrc_for_bucket(self, bucket_id: Optional[int]):
        if isinstance(self.chrc, nn.ModuleList):
            if bucket_id is None:
                return None
            return self.chrc[bucket_id % len(self.chrc)]
        return self.chrc

    def _select_chrc_module(self, x_mark_enc: Optional[torch.Tensor]):
        bucket_id = self._compute_bucket_id(x_mark_enc)
        return self._get_chrc_for_bucket(bucket_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics for monitoring."""
        stats = {
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'backbone_params': sum(p.numel() for p in self.backbone.parameters()),
            'snma_params': sum(p.numel() for p in self.snma.parameters()),
            'lora_layers': len(self.lora_layer_info),
            'is_cold_start': self._is_cold_start.item(),
        }

        if isinstance(self.chrc, nn.ModuleList):
            stats['chrc_params'] = sum(p.numel() for p in self.chrc.parameters())
            stats['memory_bank'] = [chrc.get_statistics() for chrc in self.chrc]
        elif self.chrc is not None:
            stats['chrc_params'] = sum(p.numel() for p in self.chrc.parameters())
            stats['memory_bank'] = self.chrc.get_statistics()

        return stats

    def get_config(self) -> Dict[str, Any]:
        """Get H-Mem configuration."""
        return {
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_ema_decay': self.lora_ema_decay,
            'memory_dim': self.memory_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'memory_capacity': self.memory_capacity,
            'retrieval_top_k': self.retrieval_top_k,
            'pogt_ratio': self.pogt_ratio,
            'pogt_len': self.pogt_len,
            'chrc_feature_dim': self.chrc_feature_dim,
            'use_chrc': self.use_chrc,
            'freeze_backbone': self.freeze_backbone,
            'share_pogt': self.share_pogt,
        }


def build_hmem(backbone: nn.Module, args) -> HMem:
    """
    Factory function to build H-Mem model.

    Args:
        backbone: Pre-trained backbone model
        args: Configuration arguments

    Returns:
        Initialized H-Mem model
    """
    return HMem(backbone, args)
