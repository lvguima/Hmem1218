"""
Short-term Neural Memory Adapter (SNMA) for H-Mem.

Original implementation with:
- SurpriseCalculator: Computes surprise from POGT
- NeuralMemoryState: Maintains and updates memory state
- LoRAHyperNetwork: Generates LoRA parameters from memory state

Author: H-Mem Implementation
Date: 2025-12-13
"""

import math
from typing import Optional, Tuple, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurpriseCalculator(nn.Module):
    """
    Computes surprise score from POGT via self-supervised prediction.

    Principle: Predict next step encoding → Compute prediction error → Normalize to surprise
    High surprise = Environment change = Need fast adaptation
    Low surprise = Stable environment = Slow update
    """

    def __init__(
        self,
        input_features: int,
        encoding_dim: int = 128,
        momentum: float = 0.9
    ):
        super().__init__()

        self.input_features = input_features
        self.encoding_dim = encoding_dim
        self.momentum = momentum

        # POGT encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_features, encoding_dim),
            nn.LayerNorm(encoding_dim),
            nn.GELU(),
        )

        # Predictor (self-supervised)
        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim),
            nn.GELU(),
            nn.Linear(encoding_dim, encoding_dim)
        )

        # Running statistics (for normalization)
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))

    def forward(self, pogt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pogt: [batch, pogt_len, features]

        Returns:
            surprise: [batch] surprise scores
            encoding: [batch, encoding_dim] POGT encoding
        """
        batch_size, seq_len, _ = pogt.shape

        # 1. Encode POGT (temporal pooling)
        pogt_flat = pogt.reshape(-1, self.input_features)
        encoded = self.encoder(pogt_flat)
        encoded = encoded.reshape(batch_size, seq_len, -1)

        # Mean pooling
        encoding = encoded.mean(dim=1)  # [batch, encoding_dim]

        # 2. Predict next step (self-supervised task)
        predicted = self.predictor(encoding)

        # 3. Compute prediction error (raw surprise)
        surprise_raw = (predicted - encoding).abs().sum(dim=-1)  # [batch]

        # 4. Normalize surprise
        if self.training:
            # Update running statistics without tracking gradients
            with torch.no_grad():
                batch_mean = surprise_raw.mean()
                batch_std = surprise_raw.std() + 1e-8

                self.running_mean = self.momentum * self.running_mean + \
                                   (1 - self.momentum) * batch_mean
                self.running_std = self.momentum * self.running_std + \
                                  (1 - self.momentum) * batch_std

        # Normalize to ~N(0,1)
        surprise = (surprise_raw - self.running_mean) / (self.running_std + 1e-8)
        surprise = torch.relu(surprise)  # Non-negative

        return surprise, encoding


class NeuralMemoryState(nn.Module):
    """
    Neural memory state maintenance.

    Core mechanism:
    1. Surprise modulation: Higher surprise → faster update
    2. Momentum update: Smooth memory changes
    3. Multi-head read: Flexible memory extraction
    """

    def __init__(
        self,
        memory_dim: int = 256,
        num_heads: int = 4,
        momentum: float = 0.9
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.num_heads = num_heads

        # Memory state (dynamically initialized)
        self.memory = None  # [batch, memory_dim]
        self.age = None     # [batch, 1]

        # Update gate network
        self.update_gate = nn.Sequential(
            nn.Linear(memory_dim, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Sigmoid()
        )

        # Multi-head attention for reading
        self.read_attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset memory state."""
        if device is None:
            device = next(self.parameters()).device
        self.memory = torch.zeros(batch_size, self.memory_dim, device=device)
        self.age = torch.zeros(batch_size, 1, device=device)

    def update(
        self,
        encoding: torch.Tensor,
        surprise: torch.Tensor
    ) -> torch.Tensor:
        """
        Update memory state.

        Args:
            encoding: [batch, memory_dim] new information
            surprise: [batch] surprise scores

        Returns:
            updated_memory: [batch, memory_dim]
        """
        batch_size = encoding.size(0)
        device = encoding.device

        # Initialize if needed
        if self.memory is None or self.memory.size(0) != batch_size:
            self.reset(batch_size, device)

        # Ensure memory is on correct device
        if self.memory.device != device:
            self.memory = self.memory.to(device)
            self.age = self.age.to(device)

        # Clamp surprise to a reasonable range for stability
        surprise = torch.clamp(surprise, 0.0, 1.0)

        # Guard: avoid NaN propagation
        if torch.isnan(encoding).any() or torch.isnan(surprise).any():
            return self.memory

        # 1. Compute update gate (based on current memory)
        gate = self.update_gate(self.memory)  # [batch, memory_dim]

        # 2. Surprise modulates update strength
        # surprise: [batch] → [batch, 1] → [batch, memory_dim]
        surprise_modulated = surprise.unsqueeze(-1).expand_as(gate)
        update_strength = gate * surprise_modulated

        # 3. Momentum update
        # High surprise → large update_strength → fast adaptation
        # Low surprise → small update_strength → maintain stability
        M_prev = self.memory
        M_new = (1 - update_strength) * self.memory + update_strength * encoding

        # Clamp to prevent extreme values
        M_new = torch.clamp(M_new, -10.0, 10.0)

        # Fallback if NaN occurs
        if torch.isnan(M_new).any():
            M_new = M_prev

        self.memory = M_new

        # 4. Increment memory age
        self.age = self.age + 1

        return self.memory

    def read(self, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Read memory state.

        Args:
            query: [batch, query_dim] query vector (optional)

        Returns:
            [batch, memory_dim] read memory
        """
        if query is None:
            # No query: directly return memory
            return self.memory
        else:
            # With query: use attention mechanism
            query = query.unsqueeze(1)  # [batch, 1, query_dim]
            memory = self.memory.unsqueeze(1)  # [batch, 1, memory_dim]

            attended, _ = self.read_attention(query, memory, memory)
            return attended.squeeze(1)

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get complete state."""
        return {
            'memory': self.memory,
            'age': self.age,
            'mean': self.memory.mean(dim=-1) if self.memory is not None else None,
            'std': self.memory.std(dim=-1) if self.memory is not None else None
        }

    def detach_state(self):
        """Detach memory state from the current autograd graph."""
        if self.memory is not None:
            self.memory = self.memory.detach()
        if self.age is not None:
            self.age = self.age.detach()


class LoRAHyperNetwork(nn.Module):
    """
    HyperNetwork: Generates LoRA parameters from memory state.

    Architecture:
    memory_state → shared encoder → per-layer generators → (A, B) matrices
    """

    def __init__(
        self,
        memory_dim: int = 256,
        bottleneck_dim: int = 32
    ):
        super().__init__()

        self.memory_dim = memory_dim
        self.bottleneck_dim = bottleneck_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(memory_dim, bottleneck_dim * 4),
            nn.LayerNorm(bottleneck_dim * 4),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 4, bottleneck_dim * 2),
            nn.GELU(),
        )

        # LoRA layer info (registered via register_lora_layers)
        self.lora_dims = {}
        self.generators = nn.ModuleDict()

    def register_lora_layers(self, lora_dims: Dict[str, Dict]):
        """
        Register LoRA layer information.

        Args:
            lora_dims: {
                'layer_name': {
                    'A': [rank, in_features],
                    'B': [out_features, rank],
                    'total': param_count,
                    'type': 'linear' or 'conv1d'
                }
            }
        """
        self.lora_dims = lora_dims

        # Create generator for each layer
        for layer_name, dims in lora_dims.items():
            # Sanitize name for ModuleDict
            safe_name = layer_name.replace('.', '_')

            # A matrix generator
            A_size = dims['A'][0] * dims['A'][1]
            self.generators[f'{safe_name}_A'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim * 2, A_size),
                nn.Tanh()  # Limit range
            )

            # B matrix generator - initialized to output near zero
            B_size = dims['B'][0] * dims['B'][1]
            self.generators[f'{safe_name}_B'] = nn.Sequential(
                nn.Linear(self.bottleneck_dim * 2, B_size),
                nn.Tanh()
            )

    def forward(
        self,
        memory_state: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate LoRA parameters from memory state.

        Args:
            memory_state: [batch, memory_dim] memory state

        Returns:
            lora_params: {
                'layer_name': (A, B)
                where A: [batch, rank, in_features]
                      B: [batch, out_features, rank]
            }
        """
        batch_size = memory_state.size(0)

        # 1. Shared encoding
        h = self.encoder(memory_state)  # [batch, bottleneck_dim * 2]

        # 2. Generate LoRA params for each layer
        lora_params = {}

        for layer_name, dims in self.lora_dims.items():
            safe_name = layer_name.replace('.', '_')

            # Generate A matrix
            A_flat = self.generators[f'{safe_name}_A'](h)
            A = A_flat.reshape(batch_size, dims['A'][0], dims['A'][1])

            # Generate B matrix
            B_flat = self.generators[f'{safe_name}_B'](h)
            B = B_flat.reshape(batch_size, dims['B'][0], dims['B'][1])

            lora_params[layer_name] = (A, B)

        return lora_params

    def get_param_count(self) -> Dict[str, int]:
        """Count HyperNetwork parameters."""
        total = sum(p.numel() for p in self.parameters())

        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        generator_params = sum(p.numel() for p in self.generators.parameters())

        return {
            'total': total,
            'encoder': encoder_params,
            'generators': generator_params
        }


class SNMA(nn.Module):
    """
    Short-term Neural Memory Adapter.

    Complete flow:
    POGT → Surprise Calculator → Memory Update → HyperNetwork → LoRA params
    """

    def __init__(
        self,
        input_features: int,
        memory_dim: int = 256,
        bottleneck_dim: int = 32,
        momentum: float = 0.9,
        num_heads: int = 4
    ):
        super().__init__()

        self.input_features = input_features
        self.memory_dim = memory_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoding dimension
        encoding_dim = memory_dim

        # Three main components
        self.surprise_calc = SurpriseCalculator(
            input_features,
            encoding_dim=encoding_dim,
            momentum=momentum
        )
        self.memory = NeuralMemoryState(
            memory_dim=memory_dim,
            num_heads=num_heads,
            momentum=momentum
        )
        self.hypernet = LoRAHyperNetwork(
            memory_dim=memory_dim,
            bottleneck_dim=bottleneck_dim
        )

    def register_lora_layers(self, lora_dims: Dict[str, Dict]):
        """Register backbone LoRA layers."""
        self.hypernet.register_lora_layers(lora_dims)

    def forward(
        self,
        pogt: torch.Tensor,
        return_diagnostics: bool = False
    ) -> Union[Tuple[Dict, torch.Tensor], Tuple[Dict, Dict]]:
        """
        Complete forward pass.

        Args:
            pogt: [batch, pogt_len, features]
            return_diagnostics: Whether to return diagnostic info

        Returns:
            lora_params: {layer_name: (A, B)}
            memory_state: Memory state tensor
            diagnostics: (optional) Dict with diagnostic info
        """
        # Step 1: Compute surprise
        surprise, encoding = self.surprise_calc(pogt)

        # Step 2: Update memory state
        memory_state = self.memory.update(encoding, surprise)

        # Step 3: Generate LoRA parameters
        lora_params = self.hypernet(memory_state)

        if return_diagnostics:
            diagnostics = {
                'surprise': surprise,
                'encoding': encoding,
                'memory_state': memory_state,
                'memory_stats': self.memory.get_state(),
                'lora_param_count': self.hypernet.get_param_count()
            }
            return lora_params, diagnostics

        return lora_params, memory_state

    def reset(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """Reset memory state."""
        self.memory.reset(batch_size, device)

    def detach_state(self):
        """Detach memory state to avoid graph growth between steps."""
        self.memory.detach_state()

    def get_param_stats(self) -> Dict[str, int]:
        """Get parameter statistics."""
        return {
            'surprise_calc': sum(p.numel() for p in self.surprise_calc.parameters()),
            'memory': sum(p.numel() for p in self.memory.parameters()),
            'hypernet': self.hypernet.get_param_count()['total'],
            'total': sum(p.numel() for p in self.parameters())
        }


# Keep old class names for compatibility
SurpriseCalculator = SurpriseCalculator
NeuralMemoryState = NeuralMemoryState
LoRAHyperNetwork = LoRAHyperNetwork
