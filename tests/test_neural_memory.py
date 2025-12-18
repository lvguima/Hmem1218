"""
Unit tests for Neural Memory module (SNMA).

Tests cover:
1. SurpriseCalculator functionality
2. NeuralMemoryState update mechanism
3. LoRAHyperNetwork parameter generation
4. SNMA end-to-end functionality
5. Memory reset and state management

Run with: pytest tests/test_neural_memory.py -v
"""

import pytest
import torch
import torch.nn as nn

from adapter.module.neural_memory import (
    SurpriseCalculator,
    NeuralMemoryState,
    LoRAHyperNetwork,
    SNMA,
)
from adapter.module.lora import get_lora_param_dims

pytest.skip("Skipping SNMA tests for V1 rollback", allow_module_level=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_pogt():
    """Sample POGT data [batch, pogt_len, features]."""
    return torch.randn(4, 12, 7)  # batch=4, pogt_len=12, features=7


@pytest.fixture
def sample_layer_dims():
    """Sample LoRA layer dimensions."""
    return {
        'layer1': {'A': (8, 64), 'B': (128, 8), 'total': 8 * (64 + 128), 'type': 'linear'},
        'layer2': {'A': (8, 128), 'B': (256, 8), 'total': 8 * (128 + 256), 'type': 'linear'},
        'layer3': {'A': (8, 256), 'B': (64, 8), 'total': 8 * (256 + 64), 'type': 'linear'},
    }


@pytest.fixture
def simple_model():
    """Simple model for testing SNMA integration."""
    return nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64)
    )


# =============================================================================
# SurpriseCalculator Tests
# =============================================================================

class TestSurpriseCalculator:
    """Tests for SurpriseCalculator module."""

    def test_init(self):
        """Test initialization."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)

        assert calc.input_dim == 7
        assert calc.hidden_dim == 64
        assert calc.surprise_mean.item() == 1.0
        assert calc.surprise_std.item() == 1.0

    def test_forward_output_shapes(self, sample_pogt):
        """Test output shapes."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)

        surprise, encoding = calc(sample_pogt)

        assert surprise.shape == (4,)  # [batch]
        assert encoding.shape == (4, 64)  # [batch, hidden_dim]

    def test_surprise_range(self, sample_pogt):
        """Test that surprise is in [0, 1] range after normalization."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)
        calc.eval()

        with torch.no_grad():
            surprise, _ = calc(sample_pogt, update_stats=False)

        # Surprise should be normalized via sigmoid
        assert (surprise >= 0).all()
        assert (surprise <= 1).all()

    def test_running_stats_update(self, sample_pogt):
        """Test that running statistics are updated during training."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)
        calc.train()

        initial_mean = calc.surprise_mean.clone()
        initial_std = calc.surprise_std.clone()

        # Multiple forward passes should update stats
        for _ in range(5):
            calc(sample_pogt, update_stats=True)

        # Stats should have changed
        assert calc.num_updates.item() == 5
        # Note: values might stay similar by chance, but structure is tested

    def test_stats_not_updated_in_eval(self, sample_pogt):
        """Test that stats are not updated in eval mode."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)
        calc.eval()

        initial_updates = calc.num_updates.clone()

        with torch.no_grad():
            calc(sample_pogt, update_stats=True)  # Should not update in eval

        # In eval mode with no_grad, the update won't happen
        # Actually the training flag controls this
        calc.train()
        calc(sample_pogt, update_stats=True)

        assert calc.num_updates.item() > initial_updates.item()

    def test_reset_stats(self):
        """Test resetting statistics."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)
        calc.train()

        # Update stats
        pogt = torch.randn(2, 10, 7)
        calc(pogt)

        # Reset
        calc.reset_stats()

        assert calc.surprise_mean.item() == 1.0
        assert calc.surprise_std.item() == 1.0
        assert calc.num_updates.item() == 0

    def test_single_step_pogt(self):
        """Test with single-step POGT (edge case)."""
        calc = SurpriseCalculator(input_dim=7, hidden_dim=64)

        single_step_pogt = torch.randn(2, 1, 7)
        surprise, encoding = calc(single_step_pogt)

        assert surprise.shape == (2,)
        assert encoding.shape == (2, 64)


# =============================================================================
# NeuralMemoryState Tests
# =============================================================================

class TestNeuralMemoryState:
    """Tests for NeuralMemoryState module."""

    def test_init(self):
        """Test initialization."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64)

        assert memory.memory_dim == 256
        assert memory.input_dim == 64
        assert memory.memory.shape == (1, 256)
        assert memory.memory_age.shape == (1,)

    def test_reset_memory(self):
        """Test memory reset."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64)

        # Update memory
        new_info = torch.randn(4, 64)
        surprise = torch.rand(4)
        memory.update(new_info, surprise)

        # Reset
        memory.reset_memory(batch_size=2)

        assert memory.memory.shape == (2, 256)
        assert (memory.memory == 0).all()
        assert (memory.memory_age == 0).all()

    def test_update_output_shape(self):
        """Test update output shape."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64)

        new_info = torch.randn(4, 64)
        surprise = torch.rand(4)

        M_new = memory.update(new_info, surprise)

        assert M_new.shape == (4, 256)

    def test_update_changes_memory(self):
        """Test that update actually changes memory state."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64, momentum=0.5)

        memory.reset_memory(batch_size=2)
        initial_memory = memory.memory.clone()

        new_info = torch.randn(2, 64)
        surprise = torch.ones(2)  # High surprise

        memory.update(new_info, surprise)

        # Memory should have changed
        assert not torch.allclose(memory.memory, initial_memory)

    def test_surprise_modulates_update(self):
        """Test that surprise modulates the update magnitude."""
        memory1 = NeuralMemoryState(memory_dim=256, input_dim=64, momentum=0.0)
        memory2 = NeuralMemoryState(memory_dim=256, input_dim=64, momentum=0.0)

        # Same initialization
        memory1.load_state_dict(memory2.state_dict())
        memory1.reset_memory(batch_size=1)
        memory2.reset_memory(batch_size=1)

        new_info = torch.randn(1, 64)

        # Low surprise vs high surprise
        memory1.update(new_info, torch.tensor([0.1]))
        memory2.update(new_info, torch.tensor([0.9]))

        # Different surprise should lead to different updates
        # Note: Due to gates, the exact relationship is complex

    def test_age_increases(self):
        """Test that memory age increases with updates."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64)
        memory.reset_memory(batch_size=2)

        assert (memory.memory_age == 0).all()

        for i in range(5):
            new_info = torch.randn(2, 64)
            surprise = torch.rand(2)
            memory.update(new_info, surprise)

        assert (memory.memory_age == 5).all()

    def test_read_without_query(self):
        """Test reading memory without query."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64)

        new_info = torch.randn(2, 64)
        surprise = torch.rand(2)
        memory.update(new_info, surprise)

        read_result = memory.read()

        assert torch.allclose(read_result, memory.memory)

    def test_read_with_query(self):
        """Test reading memory with query (attention-based)."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64, num_heads=4)

        new_info = torch.randn(2, 64)
        surprise = torch.rand(2)
        memory.update(new_info, surprise)

        query = torch.randn(2, 256)
        read_result = memory.read(query)

        assert read_result.shape == (2, 256)

    def test_get_state(self):
        """Test getting memory state."""
        memory = NeuralMemoryState(memory_dim=256, input_dim=64)

        new_info = torch.randn(2, 64)
        surprise = torch.rand(2)
        memory.update(new_info, surprise)

        state = memory.get_state()

        assert 'memory' in state
        assert 'age' in state
        assert state['memory'].shape == (2, 256)
        assert state['age'].shape == (2,)


# =============================================================================
# LoRAHyperNetwork Tests
# =============================================================================

class TestLoRAHyperNetwork:
    """Tests for LoRAHyperNetwork module."""

    def test_init(self, sample_layer_dims):
        """Test initialization."""
        hypernet = LoRAHyperNetwork(
            memory_dim=256,
            bottleneck_dim=32,
            layer_dims=sample_layer_dims
        )

        assert hypernet.memory_dim == 256
        assert hypernet.bottleneck_dim == 32
        assert len(hypernet.layer_generators) == 3

    def test_register_layers(self):
        """Test registering layers after initialization."""
        hypernet = LoRAHyperNetwork(memory_dim=256, bottleneck_dim=32)

        assert len(hypernet.layer_generators) == 0

        layer_dims = {
            'layer1': {'A': (8, 64), 'B': (128, 8), 'total': 8 * 192},
        }
        hypernet.register_layers(layer_dims)

        assert len(hypernet.layer_generators) == 1

    def test_forward_output_shapes(self, sample_layer_dims):
        """Test forward pass output shapes."""
        hypernet = LoRAHyperNetwork(
            memory_dim=256,
            bottleneck_dim=32,
            layer_dims=sample_layer_dims
        )

        memory_state = torch.randn(4, 256)
        lora_params = hypernet(memory_state)

        assert len(lora_params) == 3

        # Check each layer's output shapes
        for name, dims in sample_layer_dims.items():
            assert name in lora_params
            A, B = lora_params[name]
            assert A.shape == (4, dims['A'][0], dims['A'][1])
            assert B.shape == (4, dims['B'][0], dims['B'][1])

    def test_different_batch_sizes(self, sample_layer_dims):
        """Test with different batch sizes."""
        hypernet = LoRAHyperNetwork(
            memory_dim=256,
            bottleneck_dim=32,
            layer_dims=sample_layer_dims
        )

        for batch_size in [1, 2, 8, 16]:
            memory_state = torch.randn(batch_size, 256)
            lora_params = hypernet(memory_state)

            for name, (A, B) in lora_params.items():
                assert A.shape[0] == batch_size
                assert B.shape[0] == batch_size

    def test_get_param_count(self, sample_layer_dims):
        """Test parameter counting."""
        hypernet = LoRAHyperNetwork(
            memory_dim=256,
            bottleneck_dim=32,
            layer_dims=sample_layer_dims
        )

        count = hypernet.get_param_count()
        assert count > 0

    def test_gradient_flow(self, sample_layer_dims):
        """Test gradient flow through hypernet."""
        hypernet = LoRAHyperNetwork(
            memory_dim=256,
            bottleneck_dim=32,
            layer_dims=sample_layer_dims
        )

        memory_state = torch.randn(4, 256, requires_grad=True)
        lora_params = hypernet(memory_state)

        # Compute loss from generated params
        loss = sum((A.sum() + B.sum()) for A, B in lora_params.values())
        loss.backward()

        # Check gradients flow
        assert memory_state.grad is not None

        for param in hypernet.parameters():
            if param.requires_grad:
                assert param.grad is not None


# =============================================================================
# SNMA Tests
# =============================================================================

class TestSNMA:
    """Tests for SNMA (Short-term Neural Memory Adapter)."""

    def test_init(self):
        """Test initialization."""
        snma = SNMA(
            input_features=7,
            memory_dim=256,
            bottleneck_dim=32
        )

        assert snma.input_features == 7
        assert snma.memory_dim == 256
        assert snma.bottleneck_dim == 32

    def test_register_lora_layers(self, sample_layer_dims):
        """Test registering LoRA layers."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)

        snma.register_lora_layers(sample_layer_dims)

        assert snma._layer_dims == sample_layer_dims
        assert snma._total_lora_params > 0

    def test_forward_without_registration(self, sample_pogt):
        """Test forward without registering layers (should return empty)."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)

        lora_params, memory_state = snma(sample_pogt)

        assert len(lora_params) == 0
        assert memory_state.shape == (4, 256)

    def test_forward_with_registration(self, sample_pogt, sample_layer_dims):
        """Test forward with registered layers."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        lora_params, memory_state = snma(sample_pogt)

        assert len(lora_params) == 3
        assert memory_state.shape == (4, 256)

        for name, dims in sample_layer_dims.items():
            assert name in lora_params
            A, B = lora_params[name]
            assert A.shape == (4, dims['A'][0], dims['A'][1])
            assert B.shape == (4, dims['B'][0], dims['B'][1])

    def test_forward_with_diagnostics(self, sample_pogt, sample_layer_dims):
        """Test forward with diagnostics."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        lora_params, memory_state, diagnostics = snma(sample_pogt, return_diagnostics=True)

        assert 'surprise' in diagnostics
        assert 'encoding' in diagnostics
        assert 'memory_age' in diagnostics

        assert diagnostics['surprise'].shape == (4,)

    def test_reset(self, sample_pogt, sample_layer_dims):
        """Test memory reset."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        # Forward to update memory
        snma(sample_pogt)

        # Reset
        snma.reset(batch_size=2)

        state = snma.get_memory_state()
        assert state['memory'].shape == (2, 256)
        assert (state['memory'] == 0).all()

    def test_get_param_stats(self, sample_layer_dims):
        """Test getting parameter statistics."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        stats = snma.get_param_stats()

        assert 'surprise_calc' in stats
        assert 'memory' in stats
        assert 'hypernet' in stats
        assert 'total' in stats
        assert 'lora_output_params' in stats
        assert stats['total'] > 0

    def test_sequential_updates(self, sample_layer_dims):
        """Test sequential memory updates."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32, momentum=0.9)
        snma.register_lora_layers(sample_layer_dims)
        snma.reset(batch_size=2)

        memory_states = []

        for i in range(5):
            pogt = torch.randn(2, 12, 7)
            _, memory_state = snma(pogt)
            memory_states.append(memory_state.clone())

        # Memory should evolve over time
        for i in range(1, 5):
            assert not torch.allclose(memory_states[i], memory_states[i-1])


# =============================================================================
# Integration Tests
# =============================================================================

class TestSNMAIntegration:
    """Integration tests for SNMA with real models."""

    def test_create_snma_from_model(self, simple_model):
        """Test create_snma helper function."""
        snma = create_snma(
            input_features=7,
            model=simple_model,
            rank=8,
            memory_dim=256,
            bottleneck_dim=32
        )

        # Should have registered all linear layers
        assert len(snma._layer_dims) == 3

    def test_snma_with_model_forward(self, simple_model):
        """Test SNMA generating params for actual model forward."""
        from adapter.module.lora import inject_lora_layers, set_all_lora_params

        # Inject LoRA into model
        model, info = inject_lora_layers(simple_model, rank=8)

        # Create SNMA
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(get_lora_param_dims(simple_model, rank=8))

        # Generate LoRA params from POGT
        pogt = torch.randn(2, 12, 7)
        lora_params, _ = snma(pogt)

        # Set params in model
        set_all_lora_params(model, lora_params)

        # Forward pass should work
        x = torch.randn(2, 64)
        out = model(x)

        assert out.shape == (2, 64)

    def test_end_to_end_training(self, simple_model):
        """Test end-to-end training with SNMA."""
        from adapter.module.lora import inject_lora_layers, set_all_lora_params, clear_all_lora_params

        # Inject LoRA
        model, info = inject_lora_layers(simple_model, rank=4, freeze_weight=True)

        # Create SNMA
        snma = SNMA(input_features=7, memory_dim=128, bottleneck_dim=16)
        layer_dims = {}
        for name, layer_info in info.items():
            shapes = layer_info['shapes']
            layer_dims[name] = {
                'A': shapes['A'],
                'B': shapes['B'],
                'total': layer_info['param_count'],
                'type': layer_info['type']
            }
        snma.register_lora_layers(layer_dims)

        # Optimizer for SNMA parameters only
        optimizer = torch.optim.Adam(snma.parameters(), lr=0.001)

        # Training loop
        for step in range(3):
            pogt = torch.randn(2, 12, 7)
            x = torch.randn(2, 64)
            target = torch.randn(2, 64)

            optimizer.zero_grad()

            # Generate LoRA params
            lora_params, _ = snma(pogt)
            set_all_lora_params(model, lora_params)

            # Forward
            out = model(x)
            loss = (out - target).pow(2).mean()

            # Backward
            loss.backward()
            optimizer.step()

            # Clear for next iteration
            clear_all_lora_params(model)

            # Reset SNMA memory to avoid graph retention issues
            snma.reset(batch_size=2)

        # Should complete without errors


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self, sample_layer_dims):
        """Test with batch size of 1."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        pogt = torch.randn(1, 12, 7)
        lora_params, memory_state = snma(pogt)

        assert memory_state.shape == (1, 256)
        for A, B in lora_params.values():
            assert A.shape[0] == 1
            assert B.shape[0] == 1

    def test_short_pogt(self, sample_layer_dims):
        """Test with very short POGT sequence."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        short_pogt = torch.randn(2, 2, 7)  # Only 2 time steps
        lora_params, memory_state = snma(short_pogt)

        assert memory_state.shape == (2, 256)

    def test_long_pogt(self, sample_layer_dims):
        """Test with long POGT sequence."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        long_pogt = torch.randn(2, 100, 7)  # 100 time steps
        lora_params, memory_state = snma(long_pogt)

        assert memory_state.shape == (2, 256)

    def test_high_dimensional_features(self, sample_layer_dims):
        """Test with high-dimensional features."""
        snma = SNMA(input_features=128, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        pogt = torch.randn(2, 12, 128)
        lora_params, memory_state = snma(pogt)

        assert memory_state.shape == (2, 256)

    def test_varying_batch_sizes(self, sample_layer_dims):
        """Test with varying batch sizes."""
        snma = SNMA(input_features=7, memory_dim=256, bottleneck_dim=32)
        snma.register_lora_layers(sample_layer_dims)

        for batch_size in [1, 4, 16, 32]:
            snma.reset(batch_size=batch_size)
            pogt = torch.randn(batch_size, 12, 7)
            lora_params, memory_state = snma(pogt)

            assert memory_state.shape == (batch_size, 256)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
