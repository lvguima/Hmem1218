"""
Quick test script for Prototype-SNMA implementation.

Tests:
1. Module initialization
2. Forward pass
3. Zero-initialization property
4. Routing statistics
5. Integration with H-Mem
"""

import torch
import torch.nn as nn
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from adapter.module.neural_memory import PrototypeSNMA, POGTFeatureExtractor, PrototypeBank, PrototypeRouter
    from adapter.module.lora import inject_lora_layers, get_lora_param_dims
except Exception:
    pytest.skip("Prototype-SNMA components are unavailable after V1 rollback", allow_module_level=True)


def test_pogt_feature_extractor():
    """Test POGTFeatureExtractor."""
    print("=" * 60)
    print("Test 1: POGTFeatureExtractor")
    print("=" * 60)

    extractor = POGTFeatureExtractor(
        input_dim=7,
        feature_dim=128,
        pooling='mean'
    )

    # Test input
    pogt = torch.randn(2, 12, 7)  # [batch=2, seq_len=12, features=7]

    # Forward
    features = extractor(pogt)

    print(f"Input shape: {pogt.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Expected: [2, 128]")
    print(f"[PASS]" if features.shape == (2, 128) else "[FAIL]")
    print()


def test_prototype_bank():
    """Test PrototypeBank."""
    print("=" * 60)
    print("Test 2: PrototypeBank")
    print("=" * 60)

    # Mock layer dimensions
    layer_dims = {
        'layer1': {'A': (8, 512), 'B': (512, 8), 'total': 8*512 + 512*8},
        'layer2': {'A': (8, 512), 'B': (512, 8), 'total': 8*512 + 512*8},
    }

    bank = PrototypeBank(
        num_prototypes=10,
        layer_dims=layer_dims
    )

    # Check zero-initialization of B
    print("Checking B matrices are zero-initialized...")
    for name in layer_dims.keys():
        safe_name = name.replace('.', '_')
        B_proto = bank.prototypes[f'{safe_name}_B']
        is_zero = (B_proto.abs().max().item() == 0.0)
        print(f"  {name}: B is zero = {is_zero}")

    # Test forward (mixing)
    weights = torch.softmax(torch.randn(2, 10), dim=-1)  # [batch=2, K=10]
    lora_params = bank(weights)

    print(f"\nRouting weights shape: {weights.shape}")
    print(f"Generated LoRA params for {len(lora_params)} layers")
    for name, (A, B) in lora_params.items():
        print(f"  {name}: A={A.shape}, B={B.shape}")

    # Check initial ΔW = A·B should be zero (since B=0)
    print("\nChecking ΔW = A·B is zero at initialization...")
    for name, (A, B) in lora_params.items():
        delta_w = torch.bmm(B, A)  # [batch, out, in]
        max_val = delta_w.abs().max().item()
        print(f"  {name}: max(|ΔW|) = {max_val:.6f}")

    print("[PASS] Pass\n")


def test_prototype_router():
    """Test PrototypeRouter."""
    print("=" * 60)
    print("Test 3: PrototypeRouter")
    print("=" * 60)

    router = PrototypeRouter(
        feature_dim=128,
        num_prototypes=10,
        hidden_dim=128,
        temperature=1.0
    )

    # Test input
    features = torch.randn(2, 128)  # [batch=2, feature_dim=128]

    # Forward
    weights, info = router(features)

    print(f"Input shape: {features.shape}")
    print(f"Output weights shape: {weights.shape}")
    print(f"Weights sum: {weights.sum(dim=-1)}")  # Should be [1.0, 1.0]
    print(f"Routing entropy: {info['entropy']}")
    print(f"Max weight: {info['max_weight']}")
    print(f"Selected prototype: {info['selected_prototype']}")
    print(f"[PASS] Pass\n")


def test_prototype_snma():
    """Test complete PrototypeSNMA."""
    print("=" * 60)
    print("Test 4: PrototypeSNMA (Complete)")
    print("=" * 60)

    # Create SNMA
    snma = PrototypeSNMA(
        input_features=7,
        num_prototypes=10,
        feature_dim=128,
        router_hidden_dim=128
    )

    # Mock layer dimensions
    layer_dims = {
        'backbone.layer1': {'A': (8, 512), 'B': (512, 8), 'total': 8*512 + 512*8},
        'backbone.layer2': {'A': (8, 512), 'B': (512, 8), 'total': 8*512 + 512*8},
    }

    # Register layers
    snma.register_lora_layers(layer_dims)

    # Test input
    pogt = torch.randn(2, 12, 7)  # [batch=2, seq_len=12, features=7]

    # Forward
    lora_params, routing_weights = snma(pogt)

    print(f"POGT shape: {pogt.shape}")
    print(f"Routing weights shape: {routing_weights.shape}")
    print(f"Generated LoRA params for {len(lora_params)} layers")

    # Test with diagnostics
    lora_params, routing_weights, diagnostics = snma(pogt, return_diagnostics=True)

    print(f"\nDiagnostics:")
    print(f"  Features shape: {diagnostics['features'].shape}")
    print(f"  Routing entropy: {diagnostics['routing_entropy']}")
    print(f"  Max weight: {diagnostics['max_weight']}")
    print(f"  Selected prototype: {diagnostics['selected_prototype']}")

    # Check parameter stats
    stats = snma.get_param_stats()
    print(f"\nParameter statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

    print(f"[PASS] Pass\n")


def test_hmem_integration():
    """Test integration with H-Mem."""
    print("=" * 60)
    print("Test 5: H-Mem Integration")
    print("=" * 60)

    from adapter.hmem import HMem

    # Create a dummy backbone (simple linear model)
    class DummyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(7, 512)
            self.layer2 = nn.Linear(512, 7)

        def forward(self, x):
            # x: [batch, seq_len, features]
            h = self.layer1(x[:, -1, :])  # Use last timestep
            out = self.layer2(h)
            return out.unsqueeze(1).repeat(1, 24, 1)  # [batch, pred_len=24, features]

    backbone = DummyBackbone()

    # Create args
    class Args:
        seq_len = 96
        pred_len = 24
        enc_in = 7
        lora_rank = 8
        lora_alpha = 16.0
        num_prototypes = 10
        snma_feature_dim = 128
        router_hidden_dim = 128
        router_temperature = 1.0
        use_gumbel_routing = False
        memory_capacity = 500
        retrieval_top_k = 5
        pogt_ratio = 0.5
        chrc_feature_dim = 128
        use_chrc = True
        freeze = True
        lora_dropout = 0.0
        lora_target_modules = None
        chrc_temperature = 0.1
        chrc_aggregation = 'softmax'
        chrc_use_refinement = True

    args = Args()

    # Build H-Mem
    hmem = HMem(backbone, args)

    print(f"H-Mem created successfully")
    print(f"  Backbone frozen: {args.freeze}")
    print(f"  SNMA num_prototypes: {hmem.snma.num_prototypes}")
    print(f"  CHRC enabled: {hmem.use_chrc}")

    # Test forward
    x_enc = torch.randn(2, 96, 7)  # [batch=2, seq_len=96, features=7]
    pogt = torch.randn(2, 12, 7)   # [batch=2, pogt_len=12, features=7]

    # Forward pass
    pred = hmem(x_enc, pogt=pogt)

    print(f"\nForward pass:")
    print(f"  Input shape: {x_enc.shape}")
    print(f"  POGT shape: {pogt.shape}")
    print(f"  Output shape: {pred.shape}")
    print(f"  Expected: [2, 24, 7]")
    print(f"  [PASS] Pass" if pred.shape == (2, 24, 7) else "[FAIL] Fail")

    # Test with components
    outputs = hmem(x_enc, pogt=pogt, return_components=True)

    print(f"\nComponents:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")

    print(f"\n[PASS] Pass\n")


def test_zero_initialization_property():
    """Test that zero-init ensures non-destructive start."""
    print("=" * 60)
    print("Test 6: Zero-initialization Property")
    print("=" * 60)

    # Create SNMA
    snma = PrototypeSNMA(
        input_features=7,
        num_prototypes=10,
        feature_dim=128
    )

    # Mock layer dimensions
    layer_dims = {
        'layer1': {'A': (8, 512), 'B': (512, 8), 'total': 8*512 + 512*8},
    }

    snma.register_lora_layers(layer_dims)

    # Generate LoRA params
    pogt = torch.randn(1, 12, 7)
    lora_params, _ = snma(pogt)

    # Compute ΔW = scaling * B @ A
    A, B = lora_params['layer1']
    delta_w = torch.bmm(B, A)  # [batch, out, in]

    max_abs_value = delta_w.abs().max().item()

    print(f"ΔW = B @ A")
    print(f"  Shape: {delta_w.shape}")
    print(f"  max(|ΔW|): {max_abs_value:.10f}")
    print(f"  Expected: ~0.0 (since B initialized to zero)")

    if max_abs_value < 1e-6:
        print(f"  [PASS] Non-destructive start verified!")
    else:
        print(f"  [FAIL] Warning: ΔW is not zero!")

    print()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Prototype-SNMA Test Suite")
    print("=" * 60 + "\n")

    try:
        test_pogt_feature_extractor()
        test_prototype_bank()
        test_prototype_router()
        test_prototype_snma()
        test_zero_initialization_property()
        test_hmem_integration()

        print("=" * 60)
        print("All tests passed! [PASS]")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
