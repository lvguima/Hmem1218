"""
Error Memory Bank and Cross-Horizon Retrieval Corrector (CHRC) for H-Mem.

This module provides:
1. ErrorMemoryBank: Non-parametric memory storing POGT-Error pairs for retrieval
2. CHRC: Complete retrieval-based error correction module

Key features:
- Fixed capacity with intelligent eviction (LRU + importance + temporal decay)
- Efficient retrieval using cosine similarity
- Adaptive aggregation of retrieved errors
- Confidence-gated correction

Author: H-Mem Implementation
Date: 2025-12-13
"""

import math
from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class POGTFeatureEncoder(nn.Module):
    """
    Encodes POGT (Partially Observed Ground Truth) into feature vectors for retrieval.

    Uses a lightweight MLP with temporal pooling to create fixed-size representations
    regardless of POGT length.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int = 128,
        hidden_dim: Optional[int] = None,
        pooling: str = 'mean'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.pooling = pooling

        if hidden_dim is None:
            hidden_dim = feature_dim * 2

        # Temporal feature extractor
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Final projection to feature space
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

        # Optional: learnable temporal position encoding
        self.max_len = 512
        self.pos_encoding = nn.Parameter(torch.randn(1, self.max_len, hidden_dim) * 0.02)

    def forward(self, pogt: torch.Tensor) -> torch.Tensor:
        """
        Encode POGT into feature vector.

        Args:
            pogt: [batch, seq_len, features] or [batch, features]

        Returns:
            Feature vector [batch, feature_dim]
        """
        if pogt.dim() == 2:
            pogt = pogt.unsqueeze(1)  # [batch, 1, features]

        batch_size, seq_len, _ = pogt.shape

        # Encode each time step
        h = self.temporal_encoder(pogt)  # [batch, seq_len, hidden_dim]

        # Add positional encoding
        if seq_len <= self.max_len:
            h = h + self.pos_encoding[:, :seq_len, :]

        # Temporal pooling
        if self.pooling == 'mean':
            h_pooled = h.mean(dim=1)
        elif self.pooling == 'max':
            h_pooled = h.max(dim=1)[0]
        elif self.pooling == 'last':
            h_pooled = h[:, -1, :]
        else:  # attention pooling
            attn_weights = torch.softmax(h.mean(dim=-1), dim=-1)
            h_pooled = (h * attn_weights.unsqueeze(-1)).sum(dim=1)

        # Project to feature space
        features = self.projector(h_pooled)

        return features


class PredictionEncoder(nn.Module):
    """
    Encodes model predictions into feature vectors for contextual retrieval.

    Uses a lightweight MLP over the flattened prediction horizon.
    """

    def __init__(self, horizon: int, num_features: int, feature_dim: int):
        super().__init__()
        self.horizon = horizon
        self.num_features = num_features
        self.feature_dim = feature_dim

        input_dim = horizon * num_features
        hidden_dim = feature_dim * 2

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Encode prediction into feature vector.

        Args:
            prediction: [batch, horizon, num_features]

        Returns:
            Feature vector [batch, feature_dim]
        """
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(1)
        return self.encoder(prediction.flatten(1))


class ErrorMemoryBank(nn.Module):
    """
    Non-parametric memory bank storing POGT-Error pairs.

    Stores historical {POGT feature → Full horizon error} mappings
    for retrieval-augmented error correction.

    Features:
    - Fixed capacity with intelligent eviction
    - Temporal decay for old entries
    - Importance-based retention
    - Efficient batch operations

    Attributes:
        capacity: Maximum number of entries
        feature_dim: Dimension of POGT features
        horizon: Prediction horizon (error length)
        num_features: Number of time series features
    """

    def __init__(
        self,
        capacity: int = 1000,
        feature_dim: int = 128,
        horizon: int = 24,
        num_features: int = 7,
        decay_factor: float = 0.995,
        temperature: float = 0.1,
        min_entries_for_retrieval: int = 10,
        forget_decay: float = 1.0,
        forget_threshold: float = 0.0,
        max_age: int = 0
    ):
        super().__init__()

        self.capacity = capacity
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.num_features = num_features
        self.decay_factor = decay_factor
        self.temperature = temperature
        self.min_entries_for_retrieval = min_entries_for_retrieval
        self.forget_decay = forget_decay
        self.forget_threshold = forget_threshold
        self.max_age = max_age

        # Storage buffers (registered as buffers for state_dict persistence)
        self.register_buffer('keys', torch.zeros(capacity, feature_dim))
        self.register_buffer('values', torch.zeros(capacity, horizon, num_features))
        self.register_buffer('timestamps', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('access_counts', torch.zeros(capacity, dtype=torch.long))
        self.register_buffer('importance_scores', torch.ones(capacity))

        # Pointer and count
        self.register_buffer('write_pointer', torch.tensor(0, dtype=torch.long))
        self.register_buffer('num_entries', torch.tensor(0, dtype=torch.long))
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))
        self._retrieve_calls = 0
        self._log_every = 100

    @property
    def is_empty(self) -> bool:
        return self.num_entries.item() == 0

    @property
    def is_full(self) -> bool:
        return self.num_entries.item() >= self.capacity

    @property
    def current_size(self) -> int:
        return min(self.num_entries.item(), self.capacity)

    def _compute_eviction_scores(self) -> torch.Tensor:
        """
        Compute eviction scores for all entries.
        Lower score = more likely to be evicted.

        Score = importance * recency * access_frequency
        """
        n = self.current_size
        if n == 0:
            return torch.zeros(self.capacity, device=self.keys.device)

        # Recency factor: newer entries have higher scores
        age = (self.global_step - self.timestamps[:n]).float()
        max_age = age.max() + 1
        recency = 1.0 - (age / max_age)  # [0, 1], newer = higher

        # Access frequency factor
        access = self.access_counts[:n].float()
        max_access = access.max() + 1
        frequency = access / max_access  # [0, 1]

        # Importance factor (normalized)
        importance = self.importance_scores[:n]
        max_importance = importance.max() + 1e-8
        importance_norm = importance / max_importance

        # Combined score
        scores = torch.zeros(self.capacity, device=self.keys.device)
        scores[:n] = (
            0.4 * importance_norm +
            0.4 * recency +
            0.2 * frequency
        )

        return scores

    def _prune_stale_entries(self):
        """Drop entries that are too old or too low importance."""
        n = self.current_size
        if n == 0:
            return

        mask = torch.ones(n, dtype=torch.bool, device=self.keys.device)
        if self.forget_threshold > 0:
            mask = mask & (self.importance_scores[:n] >= self.forget_threshold)
        if self.max_age and self.max_age > 0:
            age = (self.global_step - self.timestamps[:n]).float()
            mask = mask & (age <= self.max_age)

        if mask.all():
            return

        keep_idx = torch.nonzero(mask, as_tuple=False).flatten()
        new_n = int(keep_idx.numel())

        if new_n > 0:
            self.keys[:new_n] = self.keys[keep_idx]
            self.values[:new_n] = self.values[keep_idx]
            self.timestamps[:new_n] = self.timestamps[keep_idx]
            self.access_counts[:new_n] = self.access_counts[keep_idx]
            self.importance_scores[:new_n] = self.importance_scores[keep_idx]

        if new_n < n:
            self.keys[new_n:n].zero_()
            self.values[new_n:n].zero_()
            self.timestamps[new_n:n].zero_()
            self.access_counts[new_n:n].zero_()
            self.importance_scores[new_n:n].zero_()

        self.num_entries.fill_(new_n)
        self.write_pointer.fill_(new_n % self.capacity)

    def _get_write_index(self) -> int:
        """Get index for writing new entry."""
        if not self.is_full:
            idx = self.write_pointer.item()
            self.write_pointer = (self.write_pointer + 1) % self.capacity
            self.num_entries = min(self.num_entries + 1, self.capacity)
            return idx
        else:
            # Evict entry with lowest score
            scores = self._compute_eviction_scores()
            return scores.argmin().item()

    def store(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ):
        """
        Store new POGT-Error pairs in memory.

        Args:
            keys: POGT features [batch, feature_dim]
            values: Full horizon errors [batch, horizon, num_features]
            importance: Optional importance scores [batch]
        """
        batch_size = keys.size(0)

        # Default importance based on error magnitude
        if importance is None:
            importance = values.abs().mean(dim=(1, 2))

        # Normalize importance
        importance = importance / (importance.max() + 1e-8)

        # Apply active forgetting to existing entries
        n = self.current_size
        if n > 0 and self.forget_decay < 1.0:
            self.importance_scores[:n] *= self.forget_decay

        for i in range(batch_size):
            idx = self._get_write_index()

            self.keys[idx] = keys[i].detach()
            self.values[idx] = values[i].detach()
            self.timestamps[idx] = self.global_step.clone()
            self.access_counts[idx] = 0
            self.importance_scores[idx] = importance[i].detach()

        self.global_step += 1

        if self.forget_threshold > 0 or (self.max_age and self.max_age > 0):
            self._prune_stale_entries()

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        min_similarity: float = 0.0,
        prev_topk_indices: Optional[torch.Tensor] = None,
        trajectory_bias: float = 0.0,
        return_indices: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        """
        Retrieve top-K similar error patterns.

        Args:
            query: Query POGT features [batch, feature_dim]
            top_k: Number of entries to retrieve
            min_similarity: Minimum similarity threshold
            prev_topk_indices: Previous top-K indices for trajectory bias
            trajectory_bias: Bias strength for successor indices
            return_indices: Whether to return top-K indices

        Returns:
            retrieved_values: [batch, top_k, horizon, num_features]
            similarities: [batch, top_k]
            valid_mask: [batch, top_k] boolean mask for valid retrievals
            top_indices: (optional) [batch, top_k]
        """
        batch_size = query.size(0)
        n = self.current_size
        device = query.device

        # Handle empty or insufficient memory
        if n < self.min_entries_for_retrieval:
            empty_indices = torch.full((batch_size, top_k), -1, device=device, dtype=torch.long)
            results = (
                torch.zeros(batch_size, top_k, self.horizon, self.num_features, device=device),
                torch.zeros(batch_size, top_k, device=device),
                torch.zeros(batch_size, top_k, dtype=torch.bool, device=device)
            )
            if return_indices:
                return results + (empty_indices,)
            return results

        # Get valid entries
        valid_keys = self.keys[:n]  # [n, feature_dim]
        valid_values = self.values[:n]  # [n, horizon, num_features]
        valid_timestamps = self.timestamps[:n]

        # Compute cosine similarity
        query_norm = F.normalize(query, p=2, dim=-1)  # [batch, feature_dim]
        keys_norm = F.normalize(valid_keys, p=2, dim=-1)  # [n, feature_dim]

        similarities = torch.mm(query_norm, keys_norm.t())  # [batch, n]

        # Apply temporal decay
        age = (self.global_step - valid_timestamps).float()
        decay = torch.pow(self.decay_factor, age)  # [n]
        similarities = similarities * decay.unsqueeze(0)  # [batch, n]

        # Apply trajectory bias for successor indices
        if trajectory_bias > 0 and prev_topk_indices is not None:
            prev_topk_indices = prev_topk_indices.to(device)
            successor_mask = torch.zeros_like(similarities)
            for b in range(batch_size):
                for idx in prev_topk_indices[b]:
                    idx_val = int(idx.item())
                    if idx_val < 0:
                        continue
                    next_idx = idx_val + 1
                    if next_idx < n:
                        successor_mask[b, next_idx] = trajectory_bias
            similarities = similarities + successor_mask

        # Get top-K
        actual_k = min(top_k, n)
        top_sims, top_indices = similarities.topk(actual_k, dim=-1)  # [batch, actual_k]

        # Gather retrieved values
        # top_indices: [batch, actual_k]
        # valid_values: [n, horizon, num_features]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, actual_k)
        retrieved = valid_values[top_indices]  # [batch, actual_k, horizon, num_features]

        # Update access counts
        for b in range(batch_size):
            for k in range(actual_k):
                idx = top_indices[b, k].item()
                self.access_counts[idx] += 1

        # Create validity mask
        valid_mask = top_sims >= min_similarity

        self._retrieve_calls += 1
        if min_similarity > 0 and self._retrieve_calls % self._log_every == 0:
            valid_ratio = valid_mask.float().mean().item()
            print(f"[CHRC] similarities: min={top_sims.min().item():.3f}, "
                  f"max={top_sims.max().item():.3f}, mean={top_sims.mean().item():.3f}, "
                  f"valid_ratio={valid_ratio:.3f}")

        # Pad if necessary
        if actual_k < top_k:
            pad_size = top_k - actual_k
            retrieved = F.pad(retrieved, (0, 0, 0, 0, 0, pad_size))
            top_sims = F.pad(top_sims, (0, pad_size))
            valid_mask = F.pad(valid_mask, (0, pad_size), value=False)
            pad_indices = torch.full((batch_size, pad_size), -1, device=device, dtype=top_indices.dtype)
            top_indices = torch.cat([top_indices, pad_indices], dim=1)

        if return_indices:
            return retrieved, top_sims, valid_mask, top_indices
        return retrieved, top_sims, valid_mask

    def aggregate(
        self,
        retrieved_values: torch.Tensor,
        similarities: torch.Tensor,
        valid_mask: torch.Tensor,
        method: str = 'weighted_mean'
    ) -> torch.Tensor:
        """
        Aggregate retrieved errors into a single correction term.

        Args:
            retrieved_values: [batch, top_k, horizon, num_features]
            similarities: [batch, top_k]
            valid_mask: [batch, top_k]
            method: 'weighted_mean', 'softmax', 'max', or 'median'

        Returns:
            Aggregated error [batch, horizon, num_features]
        """
        batch_size = retrieved_values.size(0)
        device = retrieved_values.device

        # Mask invalid entries
        masked_sims = similarities * valid_mask.float()

        # Check if any valid retrievals
        has_valid = valid_mask.any(dim=-1)  # [batch]

        if method == 'weighted_mean':
            # Simple weighted average
            weights = masked_sims / (masked_sims.sum(dim=-1, keepdim=True) + 1e-8)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # [batch, top_k, 1, 1]
            aggregated = (weights * retrieved_values).sum(dim=1)

        elif method == 'softmax':
            # Softmax-weighted average
            masked_sims_for_softmax = masked_sims.clone()
            masked_sims_for_softmax[~valid_mask] = float('-inf')
            # Handle case where all entries are masked (all -inf) to prevent NaN in softmax
            # If a row is all -inf, softmax returns NaN. We set one value to 0 to avoid this.
            # The result for these rows will be zeroed out later by 'has_valid' mask.
            all_masked = (~valid_mask).all(dim=-1)
            if all_masked.any():
                masked_sims_for_softmax[all_masked, 0] = 0.0

            weights = F.softmax(masked_sims_for_softmax / self.temperature, dim=-1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)
            aggregated = (weights * retrieved_values).sum(dim=1)

        elif method == 'max':
            # Take the most similar entry
            max_idx = masked_sims.argmax(dim=-1)  # [batch]
            aggregated = retrieved_values[torch.arange(batch_size, device=device), max_idx]

        elif method == 'median':
            # Median of valid entries (robust to outliers)
            # Mask invalid with NaN, then nanmedian
            masked_values = retrieved_values.clone()
            mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked_values)
            masked_values[~mask_expanded] = float('nan')
            aggregated = torch.nanmedian(masked_values, dim=1)[0]
            aggregated = torch.nan_to_num(aggregated, nan=0.0)

        elif method == 'adaptive':
            # Blend softmax and weighted mean based on similarity concentration
            valid_mask_f = valid_mask.float()
            valid_count = valid_mask_f.sum(dim=-1, keepdim=True)
            sum_sims = (similarities * valid_mask_f).sum(dim=-1, keepdim=True)
            mean_sims = sum_sims / (valid_count + 1e-8)
            var_sims = ((similarities - mean_sims) ** 2 * valid_mask_f).sum(dim=-1, keepdim=True)
            std_sims = torch.sqrt(var_sims / (valid_count + 1e-8) + 1e-8)
            sim_max = similarities.masked_fill(~valid_mask, float('-inf')).max(dim=-1, keepdim=True)[0]
            sim_max = torch.where(valid_count > 0, sim_max, torch.zeros_like(sim_max))

            concentration = sim_max / (std_sims + 1e-8)
            softmax_weight = torch.sigmoid(concentration - 5.0).unsqueeze(-1)

            agg_softmax = self.aggregate(retrieved_values, similarities, valid_mask, method='softmax')
            agg_mean = self.aggregate(retrieved_values, similarities, valid_mask, method='weighted_mean')
            aggregated = softmax_weight * agg_softmax + (1 - softmax_weight) * agg_mean

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Zero out batches with no valid retrievals
        aggregated = aggregated * has_valid.float().unsqueeze(-1).unsqueeze(-1)

        return aggregated

    def get_statistics(self) -> Dict[str, float]:
        """Get memory bank statistics."""
        n = self.current_size
        return {
            'num_entries': n,
            'capacity': self.capacity,
            'utilization': n / self.capacity,
            'avg_access_count': self.access_counts[:n].float().mean().item() if n > 0 else 0,
            'avg_importance': self.importance_scores[:n].mean().item() if n > 0 else 0,
            'avg_age': (self.global_step - self.timestamps[:n]).float().mean().item() if n > 0 else 0,
            'global_step': self.global_step.item()
        }

    def clear(self):
        """Clear all entries from memory bank."""
        self.keys.zero_()
        self.values.zero_()
        self.timestamps.zero_()
        self.access_counts.zero_()
        self.importance_scores.fill_(1.0)
        self.write_pointer.zero_()
        self.num_entries.zero_()
        # Don't reset global_step to maintain temporal ordering


class CHRC(nn.Module):
    """
    Cross-Horizon Retrieval Corrector.

    Complete module combining:
    - POGT feature encoding
    - Historical error retrieval from memory bank
    - Adaptive error aggregation
    - Confidence-gated correction

    This module addresses the feedback delay problem by using historical
    error patterns to predict corrections for future predictions.
    """

    def __init__(
        self,
        num_features: int,
        horizon: int,
        pogt_len: int,
        feature_dim: int = 128,
        capacity: int = 1000,
        top_k: int = 5,
        temperature: float = 0.1,
        aggregation: str = 'softmax',
        use_refinement: bool = True,
        use_dual_key: bool = True,
        use_context_key: bool = False,
        trust_threshold: float = 0.5,
        gate_steepness: float = 10.0,
        trajectory_bias: float = 0.0,
        use_error_decomp: bool = False,
        error_ema_decay: float = 0.9,
        use_horizon_mask: bool = False,
        horizon_mask_mode: str = 'exp',
        horizon_mask_decay: float = 0.98,
        horizon_mask_min: float = 0.2,
        min_similarity: float = 0.0,
        forget_decay: float = 1.0,
        forget_threshold: float = 0.0,
        max_age: int = 0
    ):
        super().__init__()

        self.num_features = num_features
        self.horizon = horizon
        self.pogt_len = pogt_len
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.aggregation = aggregation
        self.use_refinement = use_refinement
        self.use_dual_key = use_dual_key
        self.use_context_key = use_context_key
        self.trust_threshold = trust_threshold
        self.gate_steepness = gate_steepness
        self.trajectory_bias = trajectory_bias
        self.use_error_decomp = use_error_decomp
        self.error_ema_decay = error_ema_decay
        self.use_horizon_mask = use_horizon_mask
        self.horizon_mask_mode = horizon_mask_mode
        self.horizon_mask_decay = horizon_mask_decay
        self.horizon_mask_min = horizon_mask_min
        self.min_similarity = min_similarity
        self.last_topk_indices: Optional[torch.Tensor] = None
        self.error_ema: Optional[torch.Tensor] = None

        # POGT Feature Encoder
        self.pogt_encoder = POGTFeatureEncoder(
            input_dim=num_features,
            feature_dim=feature_dim,
            pooling='mean'
        )

        # Feature encoders for retrieval keys
        self.pred_encoder = None
        self.ctx_encoder = None
        fusion_components = 1  # pogt

        if self.use_dual_key:
            self.pred_encoder = PredictionEncoder(
                horizon=horizon,
                num_features=num_features,
                feature_dim=feature_dim
            )
            fusion_components += 1

        if self.use_context_key:
            self.ctx_encoder = POGTFeatureEncoder(
                input_dim=num_features,
                feature_dim=feature_dim,
                pooling='mean'
            )
            fusion_components += 1

        if fusion_components > 1:
            self.key_fusion = nn.Linear(feature_dim * fusion_components, feature_dim)
        else:
            self.key_fusion = None

        # Horizon-aware correction mask
        if self.use_horizon_mask:
            if self.horizon_mask_mode == 'learned':
                self.horizon_mask = nn.Parameter(torch.ones(horizon))
            else:
                if self.horizon_mask_mode not in ('exp', 'linear'):
                    raise ValueError("horizon_mask_mode must be 'exp', 'linear', or 'learned'")
                steps = torch.arange(horizon, dtype=torch.float)
                if self.horizon_mask_mode == 'linear':
                    denom = max(horizon - 1, 1)
                    mask = 1.0 - (steps / denom)
                else:
                    mask = torch.pow(self.horizon_mask_decay, steps)
                if self.horizon_mask_min > 0:
                    mask = torch.clamp(mask, min=self.horizon_mask_min)
                self.register_buffer('horizon_mask', mask)
        else:
            self.horizon_mask = None

        # Error Memory Bank
        self.memory_bank = ErrorMemoryBank(
            capacity=capacity,
            feature_dim=feature_dim,
            horizon=horizon,
            num_features=num_features,
            temperature=temperature,
            forget_decay=forget_decay,
            forget_threshold=forget_threshold,
            max_age=max_age
        )

        # Confidence gate: use compact summary stats to avoid high-dimensional overfitting
        gate_input_dim = feature_dim + 4  # pogt_feat + [pred_mean, pred_std, corr_mean, corr_std]
        self.confidence_gate = nn.Sequential(
            nn.Linear(gate_input_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

        # Optional: refinement network to improve retrieved correction
        if use_refinement:
            self.refiner = nn.Sequential(
                nn.Linear(horizon * num_features + feature_dim, feature_dim * 2),
                nn.LayerNorm(feature_dim * 2),
                nn.GELU(),
                nn.Linear(feature_dim * 2, horizon * num_features)
            )
        else:
            self.refiner = None

        # Retrieval quality estimator (helps with gating)
        self.quality_estimator = nn.Sequential(
            nn.Linear(top_k + feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )

    def _compute_similarity_gate(self, max_similarity: torch.Tensor) -> torch.Tensor:
        """Soft gate based on retrieval similarity."""
        return torch.sigmoid(self.gate_steepness * (max_similarity - self.trust_threshold))

    def _get_horizon_mask(self, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        """Return broadcastable horizon mask [1, H, 1] if enabled."""
        if not self.use_horizon_mask or self.horizon_mask is None:
            return None
        if isinstance(self.horizon_mask, nn.Parameter):
            mask = torch.sigmoid(self.horizon_mask)
        else:
            mask = self.horizon_mask
        return mask.to(device=device, dtype=dtype).view(1, -1, 1)

    def encode_pogt(self, pogt: torch.Tensor) -> torch.Tensor:
        """
        Encode POGT into feature vector.

        Args:
            pogt: [batch, pogt_len, num_features]

        Returns:
            [batch, feature_dim]
        """
        return self.pogt_encoder(pogt)

    def _compute_query_key(
        self,
        prediction: Optional[torch.Tensor],
        pogt: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        pogt_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute retrieval query key.

        Returns:
            query_key: [batch, feature_dim]
            pogt_features: [batch, feature_dim]
        """
        if pogt_features is None:
            pogt_features = self.encode_pogt(pogt)

        components = [pogt_features]

        if self.use_dual_key and prediction is not None and self.pred_encoder is not None:
            pred_features = self.pred_encoder(prediction)
            components.append(pred_features)

        if self.use_context_key and context is not None and self.ctx_encoder is not None:
            ctx_features = self.ctx_encoder(context)
            components.append(ctx_features)

        if len(components) == 1:
            query_key = components[0]
        else:
            fused = self.key_fusion(torch.cat(components, dim=-1))
            query_key = F.normalize(fused, p=2, dim=-1)

        return query_key, pogt_features

    def forward(
        self,
        prediction: torch.Tensor,
        pogt: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        pogt_features: Optional[torch.Tensor] = None,
        return_details: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply retrieval-based correction to prediction.

        Args:
            prediction: Model prediction [batch, horizon, num_features]
            pogt: Partially observed ground truth [batch, pogt_len, num_features]
            return_details: Whether to return detailed outputs

        Returns:
            corrected_prediction: [batch, horizon, num_features]
            details: (optional) Dict with intermediate values
        """
        batch_size = prediction.size(0)
        device = prediction.device

        # Encode POGT (or reuse shared features) and build retrieval key
        query_key, pogt_features = self._compute_query_key(
            prediction=prediction,
            pogt=pogt,
            context=context,
            pogt_features=pogt_features
        )

        # Retrieve from memory bank
        retrieved_errors, similarities, valid_mask, top_indices = self.memory_bank.retrieve(
            query_key,
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            prev_topk_indices=self.last_topk_indices if self.trajectory_bias > 0 else None,
            trajectory_bias=self.trajectory_bias,
            return_indices=True
        )
        if self.trajectory_bias > 0:
            masked_indices = top_indices.clone()
            masked_indices[~valid_mask] = -1
            self.last_topk_indices = masked_indices.detach()
        else:
            self.last_topk_indices = None

        # Aggregate retrieved errors
        aggregated_error = self.memory_bank.aggregate(
            retrieved_errors,
            similarities,
            valid_mask,
            method=self.aggregation
        )  # [batch, horizon, num_features]

        # Estimate retrieval quality
        quality_input = torch.cat([
            similarities,  # [batch, top_k]
            pogt_features  # [batch, feature_dim]
        ], dim=-1)
        retrieval_quality = self.quality_estimator(quality_input)  # [batch, 1]

        # Similarity-based soft gate
        max_similarity = similarities.max(dim=-1, keepdim=True)[0]
        similarity_gate = self._compute_similarity_gate(max_similarity)

        # Optional refinement
        if self.refiner is not None:
            refiner_input = torch.cat([
                aggregated_error.reshape(batch_size, -1),
                pogt_features
            ], dim=-1)
            refined_error = self.refiner(refiner_input)
            refined_error = refined_error.reshape(batch_size, self.horizon, self.num_features)

            # Blend refined and raw based on quality
            correction = retrieval_quality.unsqueeze(-1) * refined_error + \
                        (1 - retrieval_quality.unsqueeze(-1)) * aggregated_error
        else:
            correction = aggregated_error

        # Apply horizon-aware mask
        horizon_mask = self._get_horizon_mask(prediction.device, prediction.dtype)
        if horizon_mask is not None:
            correction = correction * horizon_mask

        # Compute confidence gate with compact stats
        pred_mean = prediction.abs().mean(dim=(1, 2))
        pred_std = prediction.std(dim=(1, 2), unbiased=False)
        corr_mean = correction.abs().mean(dim=(1, 2))
        corr_std = correction.std(dim=(1, 2), unbiased=False)
        stats = torch.stack([pred_mean, pred_std, corr_mean, corr_std], dim=-1)
        gate_input = torch.cat([pogt_features, stats], dim=-1)
        confidence = self.confidence_gate(gate_input)  # [batch, 1]

        # Modulate confidence by retrieval quality and validity
        has_valid_retrieval = valid_mask.any(dim=-1, keepdim=True).float()
        effective_confidence = confidence * retrieval_quality * similarity_gate * has_valid_retrieval

        # Apply correction
        corrected = prediction + effective_confidence.unsqueeze(-1) * correction

        if return_details:
            details = {
                'pogt_features': pogt_features,
                'retrieved_errors': retrieved_errors,
                'similarities': similarities,
                'valid_mask': valid_mask,
                'aggregated_error': aggregated_error,
                'correction': correction,
                'confidence': confidence,
                'retrieval_quality': retrieval_quality,
                'max_similarity': max_similarity,
                'similarity_gate': similarity_gate,
                'effective_confidence': effective_confidence
            }
            return corrected, details

        return corrected

    def store_error(
        self,
        pogt: torch.Tensor,
        error: torch.Tensor,
        prediction: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        importance: Optional[torch.Tensor] = None
    ):
        """
        Store observed error for future retrieval.

        Called when full ground truth becomes available (after H steps).

        Args:
            pogt: POGT at prediction time [batch, pogt_len, num_features]
            error: Full horizon error [batch, horizon, num_features]
            prediction: Prediction at the same time [batch, horizon, num_features]
            importance: Optional importance scores [batch]
        """
        # Encode POGT and compute storage key
        with torch.no_grad():
            query_key, _ = self._compute_query_key(
                prediction=prediction,
                pogt=pogt,
                context=context,
                pogt_features=None
            )

        error_to_store = error
        if self.use_error_decomp:
            with torch.no_grad():
                if (
                    self.error_ema is None or
                    self.error_ema.shape != error.shape or
                    self.error_ema.device != error.device
                ):
                    systematic = error
                else:
                    systematic = (
                        self.error_ema_decay * self.error_ema +
                        (1 - self.error_ema_decay) * error
                    )
                self.error_ema = systematic.detach()
                error_to_store = systematic

        # Store in memory bank
        self.memory_bank.store(query_key, error_to_store, importance)

    def get_statistics(self) -> Dict[str, float]:
        """Get CHRC statistics."""
        stats = self.memory_bank.get_statistics()
        stats['encoder_params'] = sum(p.numel() for p in self.pogt_encoder.parameters())
        if self.pred_encoder is not None:
            stats['pred_encoder_params'] = sum(p.numel() for p in self.pred_encoder.parameters())
        if self.ctx_encoder is not None:
            stats['ctx_encoder_params'] = sum(p.numel() for p in self.ctx_encoder.parameters())
        if self.key_fusion is not None:
            stats['key_fusion_params'] = sum(p.numel() for p in self.key_fusion.parameters())
        if isinstance(self.horizon_mask, nn.Parameter):
            stats['horizon_mask_params'] = self.horizon_mask.numel()
        stats['gate_params'] = sum(p.numel() for p in self.confidence_gate.parameters())
        if self.refiner is not None:
            stats['refiner_params'] = sum(p.numel() for p in self.refiner.parameters())
        return stats

    def reset(self):
        """Reset CHRC state (clear memory bank)."""
        self.memory_bank.clear()
        self.reset_trajectory()
        self.error_ema = None

    def reset_trajectory(self):
        """Reset trajectory state (call at sequence boundaries)."""
        self.last_topk_indices = None
