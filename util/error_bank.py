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
        min_entries_for_retrieval: int = 10
    ):
        super().__init__()

        self.capacity = capacity
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.num_features = num_features
        self.decay_factor = decay_factor
        self.temperature = temperature
        self.min_entries_for_retrieval = min_entries_for_retrieval

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

        for i in range(batch_size):
            idx = self._get_write_index()

            self.keys[idx] = keys[i].detach()
            self.values[idx] = values[i].detach()
            self.timestamps[idx] = self.global_step.clone()
            self.access_counts[idx] = 0
            self.importance_scores[idx] = importance[i].detach()

        self.global_step += 1

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        min_similarity: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve top-K similar error patterns.

        Args:
            query: Query POGT features [batch, feature_dim]
            top_k: Number of entries to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            retrieved_values: [batch, top_k, horizon, num_features]
            similarities: [batch, top_k]
            valid_mask: [batch, top_k] boolean mask for valid retrievals
        """
        batch_size = query.size(0)
        n = self.current_size
        device = query.device

        # Handle empty or insufficient memory
        if n < self.min_entries_for_retrieval:
            return (
                torch.zeros(batch_size, top_k, self.horizon, self.num_features, device=device),
                torch.zeros(batch_size, top_k, device=device),
                torch.zeros(batch_size, top_k, dtype=torch.bool, device=device)
            )

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

        # Pad if necessary
        if actual_k < top_k:
            pad_size = top_k - actual_k
            retrieved = F.pad(retrieved, (0, 0, 0, 0, 0, pad_size))
            top_sims = F.pad(top_sims, (0, pad_size))
            valid_mask = F.pad(valid_mask, (0, pad_size), value=False)

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
        min_similarity: float = 0.0
    ):
        super().__init__()

        self.num_features = num_features
        self.horizon = horizon
        self.pogt_len = pogt_len
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.aggregation = aggregation
        self.use_refinement = use_refinement
        self.min_similarity = min_similarity

        # POGT Feature Encoder
        self.pogt_encoder = POGTFeatureEncoder(
            input_dim=num_features,
            feature_dim=feature_dim,
            pooling='mean'
        )

        # Error Memory Bank
        self.memory_bank = ErrorMemoryBank(
            capacity=capacity,
            feature_dim=feature_dim,
            horizon=horizon,
            num_features=num_features,
            temperature=temperature
        )

        # Confidence gate: decides how much to trust retrieved correction
        gate_input_dim = feature_dim + horizon * num_features * 2  # pogt_feat + pred + correction
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

    def encode_pogt(self, pogt: torch.Tensor) -> torch.Tensor:
        """
        Encode POGT into feature vector.

        Args:
            pogt: [batch, pogt_len, num_features]

        Returns:
            [batch, feature_dim]
        """
        return self.pogt_encoder(pogt)

    def forward(
        self,
        prediction: torch.Tensor,
        pogt: torch.Tensor,
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

        # Encode POGT (or reuse shared features if provided)
        if pogt_features is None:
            pogt_features = self.encode_pogt(pogt)  # [batch, feature_dim]

        # Retrieve from memory bank
        retrieved_errors, similarities, valid_mask = self.memory_bank.retrieve(
            pogt_features,
            top_k=self.top_k,
            min_similarity=self.min_similarity
        )

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

        # Compute confidence gate
        gate_input = torch.cat([
            pogt_features,
            prediction.reshape(batch_size, -1),
            correction.reshape(batch_size, -1)
        ], dim=-1)
        confidence = self.confidence_gate(gate_input)  # [batch, 1]

        # Modulate confidence by retrieval quality and validity
        has_valid_retrieval = valid_mask.any(dim=-1, keepdim=True).float()
        effective_confidence = confidence * retrieval_quality * has_valid_retrieval

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
                'effective_confidence': effective_confidence
            }
            return corrected, details

        return corrected

    def store_error(
        self,
        pogt: torch.Tensor,
        error: torch.Tensor,
        importance: Optional[torch.Tensor] = None
    ):
        """
        Store observed error for future retrieval.

        Called when full ground truth becomes available (after H steps).

        Args:
            pogt: POGT at prediction time [batch, pogt_len, num_features]
            error: Full horizon error [batch, horizon, num_features]
            importance: Optional importance scores [batch]
        """
        # Encode POGT
        with torch.no_grad():
            pogt_features = self.encode_pogt(pogt)

        # Store in memory bank
        self.memory_bank.store(pogt_features, error, importance)

    def get_statistics(self) -> Dict[str, float]:
        """Get CHRC statistics."""
        stats = self.memory_bank.get_statistics()
        stats['encoder_params'] = sum(p.numel() for p in self.pogt_encoder.parameters())
        stats['gate_params'] = sum(p.numel() for p in self.confidence_gate.parameters())
        if self.refiner is not None:
            stats['refiner_params'] = sum(p.numel() for p in self.refiner.parameters())
        return stats

    def reset(self):
        """Reset CHRC state (clear memory bank)."""
        self.memory_bank.clear()
