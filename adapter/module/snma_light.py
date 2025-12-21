"""
SNMA-Light: simplified residual predictor for H-Mem.

This module avoids HyperNetwork/LoRA generation and instead predicts a residual
correction directly from POGT with a lightweight memory state.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class SNMALight(nn.Module):
    """Simplified SNMA that predicts a residual correction."""

    def __init__(
        self,
        input_features: int,
        memory_dim: int = 128,
        horizon: int = 96,
        num_features: int = 7,
    ):
        super().__init__()
        self.input_features = input_features
        self.memory_dim = memory_dim
        self.horizon = horizon
        self.num_features = num_features

        self.encoder = nn.Sequential(
            nn.Linear(input_features, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )

        self.memory_gate = nn.Linear(memory_dim * 2, memory_dim)
        self.register_buffer('memory', torch.zeros(memory_dim))

        self.residual_predictor = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim * 2),
            nn.GELU(),
            nn.Linear(memory_dim * 2, horizon * num_features),
        )

    def _update_memory(self, h: torch.Tensor) -> torch.Tensor:
        """Update and return the memory vector."""
        batch_size = h.size(0)
        mem = self.memory.unsqueeze(0).expand(batch_size, -1)
        gate = torch.sigmoid(self.memory_gate(torch.cat([h, mem], dim=-1)))

        with torch.no_grad():
            gate_mean = gate.mean(dim=0)
            h_mean = h.mean(dim=0)
            updated = gate_mean * h_mean + (1 - gate_mean) * self.memory
            self.memory.copy_(updated)

        return self.memory

    def forward(
        self,
        pogt: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            pogt: [batch, pogt_len, features]
            return_features: Whether to return the encoded POGT features

        Returns:
            residual: [batch, horizon, num_features]
            features: [batch, memory_dim] or None
        """
        if pogt.dim() == 2:
            pogt = pogt.unsqueeze(1)

        h = self.encoder(pogt.mean(dim=1))
        memory_vec = self._update_memory(h)
        memory_expanded = memory_vec.unsqueeze(0).expand(h.size(0), -1)

        pred_input = torch.cat([h, memory_expanded], dim=-1)
        residual = self.residual_predictor(pred_input)
        residual = residual.reshape(h.size(0), self.horizon, self.num_features)

        if return_features:
            return residual, h
        return residual, None

    def reset(self):
        """Reset memory state."""
        self.memory.zero_()

    def detach_state(self):
        """Detach memory state from graph (no-op for buffer)."""
        self.memory.detach_()
