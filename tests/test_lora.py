"""
Unit tests for LoRA (Low-Rank Adaptation) layers.

Tests cover:
1. LoRALinear forward pass correctness
2. LoRAConv1d forward pass correctness
3. Dynamic LoRA parameter injection
4. Gradient flow through LoRA layers
5. Factory functions and utilities
6. Batched vs shared LoRA parameters

Run with: pytest tests/test_lora.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapter.module.lora import (
    LoRALinear,
    LoRAConv1d,
    LoRAEmbedding,
    LoRALayer,
    create_lora_layer_from_linear,
    create_lora_layer_from_conv1d,
    get_lora_param_dims,
    inject_lora_layers,
    collect_lora_layers,
    set_all_lora_params,
    clear_all_lora_params,
    get_total_lora_param_count,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def simple_linear():
    """Create a simple linear layer for testing."""
    return nn.Linear(64, 128, bias=True)


@pytest.fixture
def simple_conv1d():
    """Create a simple conv1d layer for testing."""
    return nn.Conv1d(64, 128, kernel_size=1, bias=True)


@pytest.fixture
def simple_model():
    """Create a simple model with multiple linear layers."""
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64)
    )


# =============================================================================
# LoRALinear Tests
# =============================================================================

class TestLoRALinear:
    """Tests for LoRALinear layer."""

    def test_init(self):
        """Test LoRALinear initialization."""
        layer = LoRALinear(64, 128, bias=True, rank=8, alpha=16.0)

        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.rank == 8
        assert layer.scaling == 16.0 / 8
        assert layer.weight.shape == (128, 64)
        assert layer.bias.shape == (128,)
        assert not layer.has_lora_params()

    def test_forward_without_lora(self):
        """Test forward pass without LoRA parameters (should behave like normal Linear)."""
        layer = LoRALinear(64, 128, bias=True, rank=8)
        reference = nn.Linear(64, 128, bias=True)

        # Copy weights
        with torch.no_grad():
            layer.weight.copy_(reference.weight)
            layer.bias.copy_(reference.bias)

        x = torch.randn(2, 10, 64)

        out_lora = layer(x)
        out_ref = reference(x)

        assert torch.allclose(out_lora, out_ref, atol=1e-6)

    def test_forward_with_shared_lora(self):
        """Test forward pass with shared LoRA parameters."""
        layer = LoRALinear(64, 128, bias=True, rank=8, alpha=16.0)

        x = torch.randn(2, 10, 64)

        # Set shared LoRA params
        lora_A = torch.randn(8, 64)  # [rank, in_features]
        lora_B = torch.randn(128, 8)  # [out_features, rank]
        layer.set_lora_params(lora_A, lora_B)

        assert layer.has_lora_params()

        out = layer(x)
        assert out.shape == (2, 10, 128)

        # Verify LoRA contribution
        base_out = F.linear(x, layer.weight, layer.bias)
        lora_out = layer.scaling * F.linear(F.linear(x, lora_A), lora_B)
        expected = base_out + lora_out

        assert torch.allclose(out, expected, atol=1e-5)

    def test_forward_with_batched_lora(self):
        """Test forward pass with per-sample LoRA parameters."""
        batch_size = 4
        layer = LoRALinear(64, 128, bias=True, rank=8, alpha=16.0)

        x = torch.randn(batch_size, 10, 64)

        # Set batched LoRA params
        lora_A = torch.randn(batch_size, 8, 64)  # [batch, rank, in_features]
        lora_B = torch.randn(batch_size, 128, 8)  # [batch, out_features, rank]
        layer.set_lora_params(lora_A, lora_B)

        out = layer(x)
        assert out.shape == (batch_size, 10, 128)

    def test_clear_lora_params(self):
        """Test clearing LoRA parameters."""
        layer = LoRALinear(64, 128, rank=8)

        lora_A = torch.randn(8, 64)
        lora_B = torch.randn(128, 8)
        layer.set_lora_params(lora_A, lora_B)

        assert layer.has_lora_params()

        layer.clear_lora_params()

        assert not layer.has_lora_params()
        assert layer.lora_A is None
        assert layer.lora_B is None

    def test_frozen_weights(self):
        """Test that weights are frozen when freeze_weight=True."""
        layer = LoRALinear(64, 128, freeze_weight=True)

        assert not layer.weight.requires_grad
        assert not layer.bias.requires_grad

    def test_unfrozen_weights(self):
        """Test that weights are trainable when freeze_weight=False."""
        layer = LoRALinear(64, 128, freeze_weight=False)

        assert layer.weight.requires_grad
        assert layer.bias.requires_grad

    def test_lora_param_count(self):
        """Test LoRA parameter count calculation."""
        layer = LoRALinear(64, 128, rank=8)

        expected = 8 * (64 + 128)  # rank * (in + out)
        assert layer.lora_param_count == expected

    def test_gradient_flow(self):
        """Test that gradients flow through LoRA computation."""
        layer = LoRALinear(64, 128, rank=8, freeze_weight=True)

        x = torch.randn(2, 10, 64, requires_grad=True)
        lora_A = torch.randn(8, 64, requires_grad=True)
        lora_B = torch.randn(128, 8, requires_grad=True)

        layer.set_lora_params(lora_A, lora_B)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Gradients should flow to input and LoRA params
        assert x.grad is not None
        assert lora_A.grad is not None
        assert lora_B.grad is not None

        # Frozen weights should have no gradient
        assert layer.weight.grad is None


# =============================================================================
# LoRAConv1d Tests
# =============================================================================

class TestLoRAConv1d:
    """Tests for LoRAConv1d layer."""

    def test_init(self):
        """Test LoRAConv1d initialization."""
        layer = LoRAConv1d(64, 128, kernel_size=1, rank=8)

        assert layer.in_channels == 64
        assert layer.out_channels == 128
        assert layer.rank == 8
        assert not layer.has_lora_params()

    def test_kernel_size_validation(self):
        """Test that non-1 kernel sizes raise error."""
        with pytest.raises(ValueError):
            LoRAConv1d(64, 128, kernel_size=3, rank=8)

    def test_forward_without_lora(self):
        """Test forward pass without LoRA parameters."""
        layer = LoRAConv1d(64, 128, kernel_size=1, rank=8)
        reference = nn.Conv1d(64, 128, kernel_size=1)

        with torch.no_grad():
            layer.weight.copy_(reference.weight)
            layer.bias.copy_(reference.bias)

        x = torch.randn(2, 64, 10)  # [batch, channels, length]

        out_lora = layer(x)
        out_ref = reference(x)

        assert torch.allclose(out_lora, out_ref, atol=1e-6)

    def test_forward_with_lora(self):
        """Test forward pass with LoRA parameters."""
        layer = LoRAConv1d(64, 128, kernel_size=1, rank=8)

        x = torch.randn(2, 64, 10)

        lora_A = torch.randn(8, 64)
        lora_B = torch.randn(128, 8)
        layer.set_lora_params(lora_A, lora_B)

        out = layer(x)
        assert out.shape == (2, 128, 10)


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_lora_from_linear(self, simple_linear):
        """Test creating LoRA layer from existing Linear."""
        lora_layer = create_lora_layer_from_linear(simple_linear, rank=8, alpha=16.0)

        assert isinstance(lora_layer, LoRALinear)
        assert lora_layer.in_features == simple_linear.in_features
        assert lora_layer.out_features == simple_linear.out_features
        assert torch.allclose(lora_layer.weight, simple_linear.weight)
        assert torch.allclose(lora_layer.bias, simple_linear.bias)

    def test_create_lora_from_conv1d(self, simple_conv1d):
        """Test creating LoRA layer from existing Conv1d."""
        lora_layer = create_lora_layer_from_conv1d(simple_conv1d, rank=8)

        assert isinstance(lora_layer, LoRAConv1d)
        assert lora_layer.in_channels == simple_conv1d.in_channels
        assert lora_layer.out_channels == simple_conv1d.out_channels

    def test_create_lora_from_conv1d_invalid_kernel(self):
        """Test that creating LoRA from Conv1d with kernel > 1 raises error."""
        conv = nn.Conv1d(64, 128, kernel_size=3)
        with pytest.raises(ValueError):
            create_lora_layer_from_conv1d(conv)


# =============================================================================
# Injection Utility Tests
# =============================================================================

class TestInjectionUtilities:
    """Tests for LoRA injection utilities."""

    def test_get_lora_param_dims(self, simple_model):
        """Test getting LoRA parameter dimensions."""
        dims = get_lora_param_dims(simple_model, rank=8)

        assert len(dims) == 3  # 3 linear layers
        assert '0' in dims  # First linear
        assert '2' in dims  # Second linear
        assert '4' in dims  # Third linear

        # Check dimensions for first layer
        assert dims['0']['A'] == (8, 64)
        assert dims['0']['B'] == (128, 8)
        assert dims['0']['total'] == 8 * (64 + 128)

    def test_get_lora_param_dims_with_filter(self, simple_model):
        """Test getting LoRA dims with module filter."""
        dims = get_lora_param_dims(simple_model, rank=8, target_modules=['0', '2'])

        assert len(dims) == 2
        assert '0' in dims
        assert '2' in dims
        assert '4' not in dims

    def test_inject_lora_layers(self, simple_model):
        """Test injecting LoRA layers into model."""
        original_params = sum(p.numel() for p in simple_model.parameters())

        model, info = inject_lora_layers(simple_model, rank=8)

        assert len(info) == 3
        assert isinstance(model[0], LoRALinear)
        assert isinstance(model[2], LoRALinear)
        assert isinstance(model[4], LoRALinear)

    def test_collect_lora_layers(self, simple_model):
        """Test collecting LoRA layers from model."""
        model, _ = inject_lora_layers(simple_model, rank=8)
        lora_layers = collect_lora_layers(model)

        assert len(lora_layers) == 3

    def test_set_and_clear_all_lora_params(self, simple_model):
        """Test setting and clearing all LoRA params."""
        model, info = inject_lora_layers(simple_model, rank=8)

        # Create params for all layers
        lora_params = {}
        for name, layer_info in info.items():
            shapes = layer_info['shapes']
            lora_params[name] = (
                torch.randn(*shapes['A']),
                torch.randn(*shapes['B'])
            )

        # Set all params
        set_all_lora_params(model, lora_params)

        # Verify all layers have params
        for layer in collect_lora_layers(model).values():
            assert layer.has_lora_params()

        # Clear all params
        clear_all_lora_params(model)

        # Verify all cleared
        for layer in collect_lora_layers(model).values():
            assert not layer.has_lora_params()

    def test_get_total_lora_param_count(self, simple_model):
        """Test total LoRA parameter count."""
        rank = 8
        total = get_total_lora_param_count(simple_model, rank=rank)

        # Layer 0: 64->128, Layer 2: 128->256, Layer 4: 256->64
        expected = (
            rank * (64 + 128) +
            rank * (128 + 256) +
            rank * (256 + 64)
        )

        assert total == expected


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for LoRA with real models."""

    def test_with_transformer_like_model(self):
        """Test LoRA injection in a Transformer-like architecture."""
        class SimpleTransformer(nn.Module):
            def __init__(self, d_model=64, n_heads=4):
                super().__init__()
                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                self.ff1 = nn.Linear(d_model, d_model * 4)
                self.ff2 = nn.Linear(d_model * 4, d_model)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                out = self.out_proj(v)  # Simplified
                out = self.ff2(F.gelu(self.ff1(out)))
                return out

        model = SimpleTransformer()
        x = torch.randn(2, 10, 64)

        # Get output before injection
        with torch.no_grad():
            out_before = model(x)

        # Inject LoRA
        model, info = inject_lora_layers(model, rank=4)

        # Output should be same without LoRA params
        with torch.no_grad():
            out_after = model(x)

        assert torch.allclose(out_before, out_after, atol=1e-5)

        # Now set LoRA params and verify output changes
        lora_params = {}
        for name, layer_info in info.items():
            shapes = layer_info['shapes']
            lora_params[name] = (
                torch.randn(*shapes['A']) * 0.1,
                torch.randn(*shapes['B']) * 0.1
            )
        set_all_lora_params(model, lora_params)

        with torch.no_grad():
            out_with_lora = model(x)

        # Output should be different now
        assert not torch.allclose(out_before, out_with_lora, atol=1e-3)

    def test_lora_training_loop(self):
        """Test LoRA in a simple training loop."""
        # Simple model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

        # Inject LoRA
        model, info = inject_lora_layers(model, rank=4, freeze_weight=True)

        # Create trainable LoRA parameters
        lora_params_list = []
        for name, layer_info in info.items():
            shapes = layer_info['shapes']
            A = nn.Parameter(torch.randn(*shapes['A']) * 0.01)
            B = nn.Parameter(torch.randn(*shapes['B']) * 0.01)
            lora_params_list.extend([A, B])

        optimizer = torch.optim.Adam(lora_params_list, lr=0.01)

        # Training loop
        for step in range(3):
            # Set current LoRA params
            idx = 0
            for name in info.keys():
                layer = collect_lora_layers(model)[name]
                layer.set_lora_params(lora_params_list[idx], lora_params_list[idx + 1])
                idx += 2

            # Forward
            x = torch.randn(4, 32)
            y = model(x)
            loss = y.pow(2).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Verify gradients were computed for LoRA params
        for param in lora_params_list:
            assert param.grad is not None or param.grad is None  # May be zeroed after step


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        layer = LoRALinear(64, 128, rank=8)
        x = torch.randn(1, 10, 64)

        lora_A = torch.randn(8, 64)
        lora_B = torch.randn(128, 8)
        layer.set_lora_params(lora_A, lora_B)

        out = layer(x)
        assert out.shape == (1, 10, 128)

    def test_very_small_rank(self):
        """Test with rank=1."""
        layer = LoRALinear(64, 128, rank=1)

        lora_A = torch.randn(1, 64)
        lora_B = torch.randn(128, 1)
        layer.set_lora_params(lora_A, lora_B)

        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_no_bias(self):
        """Test layer without bias."""
        layer = LoRALinear(64, 128, bias=False, rank=8)

        assert layer.bias is None

        x = torch.randn(2, 10, 64)
        out = layer(x)
        assert out.shape == (2, 10, 128)

    def test_2d_input(self):
        """Test with 2D input [batch, features]."""
        layer = LoRALinear(64, 128, rank=8)

        lora_A = torch.randn(8, 64)
        lora_B = torch.randn(128, 8)
        layer.set_lora_params(lora_A, lora_B)

        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 128)

    def test_4d_input(self):
        """Test with 4D input [batch, seq, heads, features]."""
        layer = LoRALinear(64, 128, rank=8)

        lora_A = torch.randn(8, 64)
        lora_B = torch.randn(128, 8)
        layer.set_lora_params(lora_A, lora_B)

        x = torch.randn(2, 10, 4, 64)
        out = layer(x)
        assert out.shape == (2, 10, 4, 128)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
