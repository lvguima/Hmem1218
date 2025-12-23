"""
End-to-end integration tests for H-Mem.

Tests the complete H-Mem pipeline:
1. Model instantiation with real backbone
2. Two-phase training workflow
3. Online learning with delayed feedback
4. Component interaction (SNMA + CHRC)
5. State management and reset
6. Integration with OnlineTSF experiment framework

Run with: pytest tests/test_hmem_integration.py -v
"""

import pytest
pytest.skip("Legacy H-Mem (SNMA/LoRA) integration tests are disabled; runtime is CHRC-only.", allow_module_level=True)
import torch
import torch.nn as nn
import numpy as np
from argparse import Namespace

from adapter.hmem import HMem, build_hmem
from adapter.module.lora import collect_lora_layers


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleBackbone(nn.Module):
    """Simple model that outputs proper [batch, horizon, features] shape."""

    def __init__(self, input_dim=64, horizon=24, features=7):
        super().__init__()
        self.horizon = horizon
        self.features = features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, horizon * features)
        )

    def forward(self, x, x_mark=None):
        batch_size = x.size(0)
        out = self.net(x)
        return out.reshape(batch_size, self.horizon, self.features)


@pytest.fixture
def simple_backbone():
    """Simple sequential model for testing."""
    return SimpleBackbone(input_dim=64, horizon=24, features=7)


@pytest.fixture
def args():
    """H-Mem configuration arguments."""
    return Namespace(
        # Model dimensions
        seq_len=96,
        pred_len=24,
        enc_in=7,

        # LoRA settings
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
        lora_target_modules=None,

        # Neural memory settings
        memory_dim=64,
        bottleneck_dim=16,
        memory_momentum=0.9,
        memory_num_heads=4,

        # Error memory bank settings
        memory_capacity=100,
        retrieval_top_k=5,
        chrc_feature_dim=64,
        chrc_temperature=0.1,
        chrc_aggregation='softmax',
        chrc_use_refinement=True,

        # POGT settings
        pogt_ratio=0.5,

        # Training settings
        hmem_warmup_steps=10,
        hmem_joint_training=True,
        weight_decay=0.01,
        freeze=True,
        use_snma=True,
        use_chrc=True,
    )


# =============================================================================
# Basic Integration Tests
# =============================================================================

class TestHMemInstantiation:
    """Tests for H-Mem model instantiation."""

    def test_build_with_simple_backbone(self, simple_backbone, args):
        """Test building H-Mem with simple backbone."""
        hmem = build_hmem(simple_backbone, args)

        assert isinstance(hmem, HMem)
        assert hmem.seq_len == 96
        assert hmem.pred_len == 24
        assert hmem.enc_in == 7
        assert hmem.pogt_len == 12  # 0.5 * 24

    def test_lora_injection(self, simple_backbone, args):
        """Test LoRA layers are properly injected."""
        hmem = build_hmem(simple_backbone, args)

        # Check LoRA layers were injected
        assert len(hmem.lora_layer_info) > 0

        # Verify LoRA dimensions
        for layer_name, info in hmem.lora_layer_info.items():
            assert 'shapes' in info
            assert 'param_count' in info
            assert 'type' in info  # 'linear' or 'conv1d'

    def test_component_creation(self, simple_backbone, args):
        """Test all components are created."""
        hmem = build_hmem(simple_backbone, args)

        # Check SNMA exists
        assert hmem.snma is not None
        assert hasattr(hmem.snma, 'memory')
        assert hasattr(hmem.snma, 'hypernet')

        # Check CHRC exists (when enabled)
        assert hmem.chrc is not None
        assert hasattr(hmem.chrc, 'memory_bank')
        assert hasattr(hmem.chrc, 'pogt_encoder')

    def test_backbone_freezing(self, simple_backbone, args):
        """Test backbone is frozen when requested."""
        args.freeze = True
        hmem = build_hmem(simple_backbone, args)

        # Check backbone parameters are frozen
        for param in hmem.backbone.parameters():
            assert not param.requires_grad

    def test_component_disable(self, simple_backbone, args):
        """Test disabling CHRC."""
        args.use_chrc = False
        hmem = build_hmem(simple_backbone, args)

        assert hmem.chrc is None
        assert not hmem.flag_use_chrc


# =============================================================================
# Forward Pass Tests
# =============================================================================

class TestHMemForward:
    """Tests for H-Mem forward pass."""

    def test_forward_without_pogt(self, simple_backbone, args):
        """Test forward without POGT (cold start)."""
        hmem = build_hmem(simple_backbone, args)

        batch_size = 4
        x = torch.randn(batch_size, 64)

        # Forward without POGT should return base prediction
        output = hmem(x, pogt=None)

        # Check output shape
        assert output.shape == (batch_size, args.pred_len, args.enc_in)

    def test_forward_with_pogt(self, simple_backbone, args):
        """Test forward with POGT."""
        hmem = build_hmem(simple_backbone, args)

        batch_size = 4
        x = torch.randn(batch_size, 64)
        pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

        # Forward with POGT
        output = hmem(x, pogt=pogt)

        assert output.shape == (batch_size, args.pred_len, args.enc_in)

    def test_forward_with_components(self, simple_backbone, args):
        """Test forward returns all components."""
        hmem = build_hmem(simple_backbone, args)

        batch_size = 4
        x = torch.randn(batch_size, 64)
        pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

        # Forward with component details
        outputs = hmem(x, pogt=pogt, return_components=True)

        assert isinstance(outputs, dict)
        assert 'base_prediction' in outputs
        assert 'adapted_prediction' in outputs
        assert 'correction' in outputs
        assert 'prediction' in outputs
        assert 'memory_state' in outputs

    def test_gradient_flow(self, simple_backbone, args):
        """Test gradients flow through trainable components."""
        args.freeze = True  # Freeze backbone
        hmem = build_hmem(simple_backbone, args)

        batch_size = 4
        x = torch.randn(batch_size, 64)
        pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

        # Forward and backward
        output = hmem(x, pogt=pogt)
        loss = output.sum()
        loss.backward()

        # Check SNMA has gradients
        snma_has_grad = any(
            p.grad is not None
            for p in hmem.snma.parameters()
            if p.requires_grad
        )
        assert snma_has_grad

        # Check backbone has no gradients (frozen)
        backbone_has_grad = any(
            p.grad is not None
            for p in hmem.backbone.parameters()
        )
        assert not backbone_has_grad


# =============================================================================
# Memory Management Tests
# =============================================================================

class TestMemoryManagement:
    """Tests for memory state management."""

    def test_memory_reset(self, simple_backbone, args):
        """Test resetting memory states."""
        hmem = build_hmem(simple_backbone, args)

        # Make some predictions to populate memory
        for _ in range(5):
            x = torch.randn(2, 64)
            pogt = torch.randn(2, hmem.pogt_len, args.enc_in)
            output = hmem(x, pogt=pogt)

            # Simulate delayed feedback
            error = torch.randn(2, args.pred_len, args.enc_in)
            hmem.update_memory_bank(error)

        # Check memory is populated
        assert hmem.chrc.memory_bank.current_size > 0

        # Reset
        hmem.reset_memory(batch_size=2)

        # Check memory is cleared
        assert hmem.chrc.memory_bank.is_empty
        assert hmem._is_cold_start.item()

    def test_delayed_memory_update(self, simple_backbone, args):
        """Test delayed memory bank update."""
        hmem = build_hmem(simple_backbone, args)

        batch_size = 2
        x = torch.randn(batch_size, 64)
        pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

        # Forward pass stores prediction
        output = hmem(x, pogt=pogt)

        assert hmem._last_pogt is not None
        assert hmem._last_prediction is not None

        # Update memory bank when GT available
        ground_truth = torch.randn(batch_size, args.pred_len, args.enc_in)
        hmem.update_memory_bank(ground_truth)

        # Check memory bank updated
        assert hmem.chrc.memory_bank.current_size == batch_size

        # Check cold start flag updated
        assert not hmem._is_cold_start.item()

    def test_snma_memory_update(self, simple_backbone, args):
        """Test SNMA memory updates over time."""
        hmem = build_hmem(simple_backbone, args)
        hmem.snma.reset(batch_size=2)

        # Sequential updates
        for step in range(5):
            pogt = torch.randn(2, hmem.pogt_len, args.enc_in)
            x = torch.randn(2, 64)

            output = hmem(x, pogt=pogt, return_components=True)

            # Check memory state exists
            assert 'memory_state' in output
            assert output['memory_state'] is not None


# =============================================================================
# Online Learning Workflow Tests
# =============================================================================

class TestOnlineLearningWorkflow:
    """Tests simulating realistic online learning scenarios."""

    def test_cold_start_to_warm(self, simple_backbone, args):
        """Test transition from cold start to warm state."""
        hmem = build_hmem(simple_backbone, args)

        batch_size = 2

        # Cold start: no memory
        assert hmem._is_cold_start.item()
        assert hmem.chrc.memory_bank.is_empty

        # First predictions (cold start)
        x = torch.randn(batch_size, 64)
        pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

        output_cold = hmem(x, pogt=pogt, return_components=True)

        # Store several error patterns
        for i in range(10):
            x = torch.randn(batch_size, 64)
            pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

            output = hmem(x, pogt=pogt)

            # Simulate delayed feedback
            gt = torch.randn(batch_size, args.pred_len, args.enc_in)
            hmem.update_memory_bank(gt)

        # Now warm: memory populated
        assert not hmem._is_cold_start.item()
        assert hmem.chrc.memory_bank.current_size > 0

        # Warm prediction should use CHRC
        output_warm = hmem(x, pogt=pogt, return_components=True)

        # In warm state, should have non-zero correction
        if hmem.chrc.memory_bank.current_size >= hmem.chrc.memory_bank.min_entries_for_retrieval:
            # May have correction applied
            pass

    def test_sequential_online_updates(self, simple_backbone, args):
        """Test sequential online learning updates."""
        hmem = build_hmem(simple_backbone, args)
        optimizer = torch.optim.AdamW(
            [p for p in hmem.parameters() if p.requires_grad],
            lr=0.001
        )

        batch_size = 2
        num_steps = 20

        losses = []

        for step in range(num_steps):
            # Generate synthetic streaming data
            x = torch.randn(batch_size, 64)
            pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)
            ground_truth = torch.randn(batch_size, args.pred_len, args.enc_in)

            # Forward
            optimizer.zero_grad()
            output = hmem(x, pogt=pogt)

            # Compute loss
            loss = (output - ground_truth).pow(2).mean()
            losses.append(loss.item())

            # Backward
            loss.backward()
            optimizer.step()

            # Reset SNMA memory to avoid graph retention
            hmem.snma.reset(batch_size=batch_size)

            # Delayed memory update (simulate H steps later)
            if step >= args.pred_len:
                hmem.update_memory_bank(ground_truth)

        # Check training progressed
        assert len(losses) == num_steps

        # Check memory bank populated
        expected_size = min(num_steps - args.pred_len, args.memory_capacity)
        if expected_size > 0:
            assert hmem.chrc.memory_bank.current_size > 0

    def test_two_phase_training(self, simple_backbone, args):
        """Test two-phase training (warmup SNMA → joint training)."""
        args.hmem_warmup_steps = 5
        hmem = build_hmem(simple_backbone, args)

        optimizer = torch.optim.AdamW(
            [p for p in hmem.parameters() if p.requires_grad],
            lr=0.001
        )

        batch_size = 2

        # Phase 1: Warmup SNMA only
        hmem.freeze_chrc(True)

        for step in range(args.hmem_warmup_steps):
            x = torch.randn(batch_size, 64)
            pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)
            gt = torch.randn(batch_size, args.pred_len, args.enc_in)

            optimizer.zero_grad()
            output = hmem(x, pogt=pogt)
            loss = (output - gt).pow(2).mean()
            loss.backward()
            optimizer.step()

            # Reset SNMA memory to avoid graph retention
            hmem.snma.reset(batch_size=batch_size)

        # Phase 2: Joint training
        hmem.freeze_chrc(False)

        for step in range(10):
            x = torch.randn(batch_size, 64)
            pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)
            gt = torch.randn(batch_size, args.pred_len, args.enc_in)

            optimizer.zero_grad()
            output = hmem(x, pogt=pogt)
            loss = (output - gt).pow(2).mean()
            loss.backward()
            optimizer.step()

            # Reset SNMA memory to avoid graph retention
            hmem.snma.reset(batch_size=batch_size)

            # Update memory bank
            hmem.update_memory_bank(gt)


# =============================================================================
# Component Control Tests
# =============================================================================

class TestComponentControl:
    """Tests for enabling/disabling components."""

    def test_disable_snma(self, simple_backbone, args):
        """Test disabling SNMA."""
        hmem = build_hmem(simple_backbone, args)
        hmem.enable_snma(False)

        assert not hmem.flag_use_snma

        # Forward should skip SNMA
        x = torch.randn(2, 64)
        pogt = torch.randn(2, hmem.pogt_len, args.enc_in)

        output = hmem(x, pogt=pogt, return_components=True)

        # Adapted prediction should equal base prediction
        assert torch.allclose(
            output['adapted_prediction'],
            output['base_prediction']
        )

    def test_disable_chrc(self, simple_backbone, args):
        """Test disabling CHRC."""
        hmem = build_hmem(simple_backbone, args)
        hmem.enable_chrc(False)

        assert not hmem.flag_use_chrc

        # Populate memory first
        for _ in range(10):
            x = torch.randn(2, 64)
            pogt = torch.randn(2, hmem.pogt_len, args.enc_in)
            output = hmem(x, pogt=pogt)
            gt = torch.randn(2, args.pred_len, args.enc_in)
            hmem.update_memory_bank(gt)

        # Forward should skip CHRC even with populated memory
        x = torch.randn(2, 64)
        pogt = torch.randn(2, hmem.pogt_len, args.enc_in)

        output = hmem(x, pogt=pogt, return_components=True)

        # Correction should be zero
        assert torch.allclose(
            output['correction'],
            torch.zeros_like(output['correction'])
        )

    def test_freeze_unfreeze_components(self, simple_backbone, args):
        """Test freezing and unfreezing components."""
        hmem = build_hmem(simple_backbone, args)

        # Freeze SNMA
        hmem.freeze_snma(True)
        assert all(not p.requires_grad for p in hmem.snma.parameters())

        # Unfreeze SNMA
        hmem.freeze_snma(False)
        assert any(p.requires_grad for p in hmem.snma.parameters())

        # Freeze CHRC
        hmem.freeze_chrc(True)
        assert all(not p.requires_grad for p in hmem.chrc.parameters())

        # Unfreeze CHRC
        hmem.freeze_chrc(False)
        assert any(p.requires_grad for p in hmem.chrc.parameters())


# =============================================================================
# Statistics and Monitoring Tests
# =============================================================================

class TestStatistics:
    """Tests for model statistics."""

    def test_get_statistics(self, simple_backbone, args):
        """Test statistics retrieval."""
        hmem = build_hmem(simple_backbone, args)

        stats = hmem.get_statistics()

        assert 'total_params' in stats
        assert 'trainable_params' in stats
        assert 'backbone_params' in stats
        assert 'snma_params' in stats
        assert 'chrc_params' in stats
        assert 'lora_layers' in stats
        assert 'is_cold_start' in stats
        assert 'memory_bank' in stats

    def test_get_config(self, simple_backbone, args):
        """Test configuration retrieval."""
        hmem = build_hmem(simple_backbone, args)

        config = hmem.get_config()

        assert config['lora_rank'] == args.lora_rank
        assert config['memory_dim'] == args.memory_dim
        assert config['pogt_ratio'] == args.pogt_ratio
        assert config['pogt_len'] == hmem.pogt_len
        assert config['use_chrc'] == args.use_chrc


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample_batch(self, simple_backbone, args):
        """Test with batch size of 1."""
        hmem = build_hmem(simple_backbone, args)

        x = torch.randn(1, 64)
        pogt = torch.randn(1, hmem.pogt_len, args.enc_in)

        output = hmem(x, pogt=pogt)
        assert output.shape == (1, args.pred_len, args.enc_in)

    def test_large_batch(self, simple_backbone, args):
        """Test with large batch size."""
        hmem = build_hmem(simple_backbone, args)

        batch_size = 64
        x = torch.randn(batch_size, 64)
        pogt = torch.randn(batch_size, hmem.pogt_len, args.enc_in)

        output = hmem(x, pogt=pogt)
        assert output.shape == (batch_size, args.pred_len, args.enc_in)

    def test_different_pogt_ratios(self, simple_backbone, args):
        """Test different POGT ratios."""
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            args.pogt_ratio = ratio
            hmem = build_hmem(simple_backbone, args)

            expected_len = max(1, int(args.pred_len * ratio))
            assert hmem.pogt_len == expected_len

    def test_zero_capacity_memory_bank(self, simple_backbone, args):
        """Test CHRC with minimal capacity."""
        args.memory_capacity = 10
        args.retrieval_top_k = 3

        hmem = build_hmem(simple_backbone, args)

        # Fill beyond capacity
        for _ in range(20):
            x = torch.randn(2, 64)
            pogt = torch.randn(2, hmem.pogt_len, args.enc_in)
            output = hmem(x, pogt=pogt)
            gt = torch.randn(2, args.pred_len, args.enc_in)
            hmem.update_memory_bank(gt)

        # Should be at capacity, not exceed
        assert hmem.chrc.memory_bank.current_size == args.memory_capacity


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
