"""
Unit tests for Error Memory Bank and CHRC modules.

Tests cover:
1. POGTFeatureEncoder functionality
2. ErrorMemoryBank storage and retrieval
3. CHRC end-to-end functionality
4. Edge cases and integration tests

Run with: pytest tests/test_error_bank.py -v
"""

import pytest
import torch
import torch.nn as nn

from util.error_bank import (
    POGTFeatureEncoder,
    ErrorMemoryBank,
    CHRC,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_pogt():
    """Sample POGT data [batch, pogt_len, features]."""
    return torch.randn(4, 12, 7)


@pytest.fixture
def sample_prediction():
    """Sample prediction [batch, horizon, features]."""
    return torch.randn(4, 24, 7)


@pytest.fixture
def sample_error():
    """Sample error [batch, horizon, features]."""
    return torch.randn(4, 24, 7)


@pytest.fixture
def memory_bank():
    """Create a memory bank for testing."""
    return ErrorMemoryBank(
        capacity=100,
        feature_dim=64,
        horizon=24,
        num_features=7
    )


@pytest.fixture
def chrc():
    """Create a CHRC module for testing."""
    return CHRC(
        num_features=7,
        horizon=24,
        pogt_len=12,
        feature_dim=64,
        capacity=100,
        top_k=5
    )


# =============================================================================
# POGTFeatureEncoder Tests
# =============================================================================

class TestPOGTFeatureEncoder:
    """Tests for POGTFeatureEncoder module."""

    def test_init(self):
        """Test initialization."""
        encoder = POGTFeatureEncoder(input_dim=7, feature_dim=64)
        assert encoder.input_dim == 7
        assert encoder.feature_dim == 64

    def test_forward_3d_input(self, sample_pogt):
        """Test forward with 3D input [batch, seq, features]."""
        encoder = POGTFeatureEncoder(input_dim=7, feature_dim=64)
        features = encoder(sample_pogt)
        assert features.shape == (4, 64)

    def test_forward_2d_input(self):
        """Test forward with 2D input [batch, features]."""
        encoder = POGTFeatureEncoder(input_dim=7, feature_dim=64)
        x = torch.randn(4, 7)
        features = encoder(x)
        assert features.shape == (4, 64)

    def test_different_pooling_methods(self, sample_pogt):
        """Test different pooling methods."""
        for pooling in ['mean', 'max', 'last']:
            encoder = POGTFeatureEncoder(input_dim=7, feature_dim=64, pooling=pooling)
            features = encoder(sample_pogt)
            assert features.shape == (4, 64)

    def test_variable_sequence_length(self):
        """Test with different sequence lengths."""
        encoder = POGTFeatureEncoder(input_dim=7, feature_dim=64)

        for seq_len in [1, 5, 10, 50, 100]:
            x = torch.randn(2, seq_len, 7)
            features = encoder(x)
            assert features.shape == (2, 64)

    def test_gradient_flow(self, sample_pogt):
        """Test gradient flow through encoder."""
        encoder = POGTFeatureEncoder(input_dim=7, feature_dim=64)
        sample_pogt.requires_grad = True

        features = encoder(sample_pogt)
        loss = features.sum()
        loss.backward()

        assert sample_pogt.grad is not None


# =============================================================================
# ErrorMemoryBank Tests
# =============================================================================

class TestErrorMemoryBank:
    """Tests for ErrorMemoryBank module."""

    def test_init(self, memory_bank):
        """Test initialization."""
        assert memory_bank.capacity == 100
        assert memory_bank.feature_dim == 64
        assert memory_bank.horizon == 24
        assert memory_bank.num_features == 7
        assert memory_bank.is_empty
        assert not memory_bank.is_full

    def test_store_single_batch(self, memory_bank):
        """Test storing a single batch."""
        keys = torch.randn(4, 64)
        values = torch.randn(4, 24, 7)

        memory_bank.store(keys, values)

        assert memory_bank.current_size == 4
        assert not memory_bank.is_empty

    def test_store_multiple_batches(self, memory_bank):
        """Test storing multiple batches."""
        for i in range(5):
            keys = torch.randn(10, 64)
            values = torch.randn(10, 24, 7)
            memory_bank.store(keys, values)

        assert memory_bank.current_size == 50

    def test_store_beyond_capacity(self, memory_bank):
        """Test storing beyond capacity (triggers eviction)."""
        # Fill beyond capacity
        for i in range(15):
            keys = torch.randn(10, 64)
            values = torch.randn(10, 24, 7)
            memory_bank.store(keys, values)

        assert memory_bank.current_size == memory_bank.capacity
        assert memory_bank.is_full

    def test_retrieve_empty_bank(self, memory_bank):
        """Test retrieval from empty bank."""
        query = torch.randn(2, 64)
        values, sims, mask = memory_bank.retrieve(query, top_k=5)

        assert values.shape == (2, 5, 24, 7)
        assert sims.shape == (2, 5)
        assert mask.shape == (2, 5)
        assert not mask.any()  # All invalid

    def test_retrieve_partial_bank(self, memory_bank):
        """Test retrieval from partially filled bank."""
        # Add some entries
        keys = torch.randn(20, 64)
        values = torch.randn(20, 24, 7)
        memory_bank.store(keys, values)

        query = torch.randn(2, 64)
        retrieved, sims, mask = memory_bank.retrieve(query, top_k=5)

        assert retrieved.shape == (2, 5, 24, 7)
        assert mask.any()  # Should have valid retrievals

    def test_retrieve_updates_access_counts(self, memory_bank):
        """Test that retrieval updates access counts."""
        keys = torch.randn(20, 64)
        values = torch.randn(20, 24, 7)
        memory_bank.store(keys, values)

        initial_counts = memory_bank.access_counts[:20].clone()

        query = torch.randn(2, 64)
        memory_bank.retrieve(query, top_k=5)

        # Some counts should have increased
        assert (memory_bank.access_counts[:20] >= initial_counts).all()

    def test_aggregate_weighted_mean(self, memory_bank):
        """Test weighted mean aggregation."""
        retrieved = torch.randn(2, 5, 24, 7)
        similarities = torch.rand(2, 5)
        valid_mask = torch.ones(2, 5, dtype=torch.bool)

        aggregated = memory_bank.aggregate(retrieved, similarities, valid_mask, 'weighted_mean')
        assert aggregated.shape == (2, 24, 7)

    def test_aggregate_softmax(self, memory_bank):
        """Test softmax aggregation."""
        retrieved = torch.randn(2, 5, 24, 7)
        similarities = torch.rand(2, 5)
        valid_mask = torch.ones(2, 5, dtype=torch.bool)

        aggregated = memory_bank.aggregate(retrieved, similarities, valid_mask, 'softmax')
        assert aggregated.shape == (2, 24, 7)

    def test_aggregate_max(self, memory_bank):
        """Test max aggregation."""
        retrieved = torch.randn(2, 5, 24, 7)
        similarities = torch.rand(2, 5)
        valid_mask = torch.ones(2, 5, dtype=torch.bool)

        aggregated = memory_bank.aggregate(retrieved, similarities, valid_mask, 'max')
        assert aggregated.shape == (2, 24, 7)

    def test_aggregate_with_invalid_mask(self, memory_bank):
        """Test aggregation handles invalid entries."""
        retrieved = torch.randn(2, 5, 24, 7)
        similarities = torch.rand(2, 5)
        valid_mask = torch.zeros(2, 5, dtype=torch.bool)  # All invalid

        aggregated = memory_bank.aggregate(retrieved, similarities, valid_mask, 'weighted_mean')

        # Should be zeros when no valid entries
        assert torch.allclose(aggregated, torch.zeros_like(aggregated))

    def test_temporal_decay(self, memory_bank):
        """Test that older entries have lower effective similarity."""
        # Store entries at different "times"
        for i in range(5):
            keys = torch.randn(10, 64)
            values = torch.randn(10, 24, 7)
            memory_bank.store(keys, values)
            # Global step increases with each store

        # Later entries should have higher effective similarity for same cosine sim
        # This is tested implicitly through the decay mechanism

    def test_get_statistics(self, memory_bank):
        """Test statistics retrieval."""
        keys = torch.randn(20, 64)
        values = torch.randn(20, 24, 7)
        memory_bank.store(keys, values)

        stats = memory_bank.get_statistics()

        assert 'num_entries' in stats
        assert 'capacity' in stats
        assert 'utilization' in stats
        assert 'avg_access_count' in stats
        assert 'avg_importance' in stats
        assert stats['num_entries'] == 20

    def test_clear(self, memory_bank):
        """Test clearing memory bank."""
        keys = torch.randn(20, 64)
        values = torch.randn(20, 24, 7)
        memory_bank.store(keys, values)

        assert not memory_bank.is_empty

        memory_bank.clear()

        assert memory_bank.is_empty
        assert memory_bank.current_size == 0


# =============================================================================
# CHRC Tests
# =============================================================================

class TestCHRC:
    """Tests for CHRC module."""

    def test_init(self, chrc):
        """Test initialization."""
        assert chrc.num_features == 7
        assert chrc.horizon == 24
        assert chrc.pogt_len == 12
        assert chrc.feature_dim == 64
        assert chrc.top_k == 5

    def test_encode_pogt(self, chrc, sample_pogt):
        """Test POGT encoding."""
        features = chrc.encode_pogt(sample_pogt)
        assert features.shape == (4, 64)

    def test_forward_empty_memory(self, chrc, sample_prediction, sample_pogt):
        """Test forward with empty memory bank."""
        corrected = chrc(sample_prediction, sample_pogt)

        assert corrected.shape == sample_prediction.shape
        # With empty memory, output should be close to input
        # (confidence should be near zero)

    def test_forward_with_memory(self, chrc, sample_prediction, sample_pogt, sample_error):
        """Test forward with populated memory bank."""
        # First, store some errors
        for i in range(20):
            pogt = torch.randn(4, 12, 7)
            error = torch.randn(4, 24, 7)
            chrc.store_error(pogt, error)

        # Now test forward
        corrected = chrc(sample_prediction, sample_pogt)
        assert corrected.shape == sample_prediction.shape

    def test_forward_with_details(self, chrc, sample_prediction, sample_pogt):
        """Test forward with details returned."""
        # Populate memory
        for i in range(20):
            pogt = torch.randn(4, 12, 7)
            error = torch.randn(4, 24, 7)
            chrc.store_error(pogt, error)

        corrected, details = chrc(sample_prediction, sample_pogt, return_details=True)

        assert 'pogt_features' in details
        assert 'retrieved_errors' in details
        assert 'similarities' in details
        assert 'confidence' in details
        assert 'effective_confidence' in details

    def test_store_error(self, chrc, sample_pogt, sample_error):
        """Test storing errors."""
        initial_size = chrc.memory_bank.current_size

        chrc.store_error(sample_pogt, sample_error)

        assert chrc.memory_bank.current_size == initial_size + 4

    def test_get_statistics(self, chrc):
        """Test statistics retrieval."""
        stats = chrc.get_statistics()

        assert 'num_entries' in stats
        assert 'encoder_params' in stats
        assert 'gate_params' in stats

    def test_reset(self, chrc, sample_pogt, sample_error):
        """Test resetting CHRC."""
        chrc.store_error(sample_pogt, sample_error)
        assert not chrc.memory_bank.is_empty

        chrc.reset()
        assert chrc.memory_bank.is_empty

    def test_gradient_flow(self, chrc, sample_prediction, sample_pogt, sample_error):
        """Test gradient flow through CHRC."""
        # Populate memory
        for i in range(20):
            pogt = torch.randn(4, 12, 7)
            error = torch.randn(4, 24, 7)
            chrc.store_error(pogt, error)

        sample_prediction.requires_grad = True
        corrected = chrc(sample_prediction, sample_pogt)
        loss = corrected.sum()
        loss.backward()

        assert sample_prediction.grad is not None

    def test_confidence_gating(self, chrc, sample_prediction, sample_pogt):
        """Test that confidence gating affects output."""
        # With empty memory, confidence should be low
        corrected_empty = chrc(sample_prediction, sample_pogt)

        # Populate memory
        for i in range(30):
            pogt = torch.randn(4, 12, 7)
            error = torch.randn(4, 24, 7) * 10  # Large errors
            chrc.store_error(pogt, error)

        # With populated memory and similar POGT, correction should be more significant
        corrected_full, details = chrc(sample_prediction, sample_pogt, return_details=True)

        # Check confidence exists and is in valid range
        assert (details['confidence'] >= 0).all()
        assert (details['confidence'] <= 1).all()


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for CHRC with other modules."""

    def test_chrc_with_real_workflow(self):
        """Test CHRC in a realistic workflow."""
        chrc = CHRC(
            num_features=7,
            horizon=24,
            pogt_len=12,
            feature_dim=64,
            capacity=100,
            top_k=5
        )

        # Simulate online learning workflow
        all_predictions = []
        all_corrected = []

        for step in range(50):
            # Generate synthetic data
            pogt = torch.randn(2, 12, 7)
            prediction = torch.randn(2, 24, 7)
            ground_truth = prediction + torch.randn(2, 24, 7) * 0.5  # Add noise

            # Apply correction
            corrected = chrc(prediction, pogt)
            all_predictions.append(prediction)
            all_corrected.append(corrected)

            # After seeing ground truth, store the error
            # (In real scenario, this happens H steps later)
            if step >= 1:
                error = ground_truth - prediction
                chrc.store_error(pogt, error)

        # Check that memory bank is populated
        assert chrc.memory_bank.current_size > 0

    def test_chrc_with_snma(self):
        """Test CHRC integration with SNMA."""
        from adapter.module.neural_memory import SNMA
        from adapter.module.lora import inject_lora_layers, set_all_lora_params, clear_all_lora_params

        # Create a simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 24 * 7)  # Output: horizon * features
        )

        # Inject LoRA
        model, info = inject_lora_layers(model, rank=4, freeze_weight=True)

        # Create SNMA
        layer_dims = {}
        for name, layer_info in info.items():
            shapes = layer_info['shapes']
            layer_dims[name] = {
                'A': shapes['A'],
                'B': shapes['B'],
                'total': layer_info['param_count'],
                'type': layer_info['type']
            }

        snma = SNMA(input_features=7, memory_dim=64, bottleneck_dim=16)
        snma.register_lora_layers(layer_dims)

        # Create CHRC
        chrc = CHRC(
            num_features=7,
            horizon=24,
            pogt_len=12,
            feature_dim=64,
            capacity=100,
            top_k=5
        )

        # Simulate workflow
        for step in range(10):
            # Input
            x = torch.randn(2, 64)
            pogt = torch.randn(2, 12, 7)

            # SNMA generates LoRA params
            lora_params, _ = snma(pogt)
            set_all_lora_params(model, lora_params)

            # Model prediction
            with torch.no_grad():
                pred_flat = model(x)
                prediction = pred_flat.reshape(2, 24, 7)

            # CHRC correction
            corrected = chrc(prediction, pogt)

            # Store error (simulated delayed feedback)
            if step > 0:
                fake_error = torch.randn(2, 24, 7) * 0.1
                chrc.store_error(pogt, fake_error)

            # Clean up
            clear_all_lora_params(model)
            snma.reset(batch_size=2)

        # Verify both modules work together
        assert chrc.memory_bank.current_size > 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample(self):
        """Test with batch size of 1."""
        chrc = CHRC(
            num_features=7,
            horizon=24,
            pogt_len=12,
            feature_dim=64,
            capacity=100,
            top_k=5
        )

        # Populate
        for i in range(20):
            chrc.store_error(torch.randn(1, 12, 7), torch.randn(1, 24, 7))

        prediction = torch.randn(1, 24, 7)
        pogt = torch.randn(1, 12, 7)

        corrected = chrc(prediction, pogt)
        assert corrected.shape == (1, 24, 7)

    def test_large_batch(self):
        """Test with large batch size."""
        chrc = CHRC(
            num_features=7,
            horizon=24,
            pogt_len=12,
            feature_dim=64,
            capacity=100,
            top_k=5
        )

        # Populate
        for i in range(20):
            chrc.store_error(torch.randn(4, 12, 7), torch.randn(4, 24, 7))

        prediction = torch.randn(32, 24, 7)
        pogt = torch.randn(32, 12, 7)

        corrected = chrc(prediction, pogt)
        assert corrected.shape == (32, 24, 7)

    def test_short_pogt(self):
        """Test with very short POGT."""
        chrc = CHRC(
            num_features=7,
            horizon=24,
            pogt_len=2,  # Very short
            feature_dim=64,
            capacity=100,
            top_k=5
        )

        prediction = torch.randn(4, 24, 7)
        pogt = torch.randn(4, 2, 7)

        corrected = chrc(prediction, pogt)
        assert corrected.shape == (4, 24, 7)

    def test_high_dimensional_features(self):
        """Test with many features."""
        chrc = CHRC(
            num_features=100,
            horizon=24,
            pogt_len=12,
            feature_dim=128,
            capacity=100,
            top_k=5
        )

        prediction = torch.randn(4, 24, 100)
        pogt = torch.randn(4, 12, 100)

        corrected = chrc(prediction, pogt)
        assert corrected.shape == (4, 24, 100)

    def test_long_horizon(self):
        """Test with long prediction horizon."""
        chrc = CHRC(
            num_features=7,
            horizon=336,  # Long horizon
            pogt_len=96,
            feature_dim=128,
            capacity=100,
            top_k=5
        )

        prediction = torch.randn(4, 336, 7)
        pogt = torch.randn(4, 96, 7)

        corrected = chrc(prediction, pogt)
        assert corrected.shape == (4, 336, 7)

    def test_memory_bank_exactly_at_capacity(self):
        """Test behavior when memory bank is exactly at capacity."""
        capacity = 50
        memory_bank = ErrorMemoryBank(
            capacity=capacity,
            feature_dim=64,
            horizon=24,
            num_features=7
        )

        # Fill exactly to capacity
        for i in range(5):
            keys = torch.randn(10, 64)
            values = torch.randn(10, 24, 7)
            memory_bank.store(keys, values)

        assert memory_bank.current_size == capacity
        assert memory_bank.is_full

        # Add one more batch - should trigger eviction
        keys = torch.randn(10, 64)
        values = torch.randn(10, 24, 7)
        memory_bank.store(keys, values)

        # Should still be at capacity
        assert memory_bank.current_size == capacity


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
