"""
Test advanced fusion strategies.
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("Testing Advanced Fusion Strategies")
print("="*70)

# Import fusion modules
from medtabx.interaction.cross_modal_fusion import CrossModalAttentionFusion
from medtabx.interaction.gated_fusion import GatedFusion
from medtabx.interaction.hierarchical_fusion import HierarchicalFusion

# Simulate data
batch_size = 8
channels = 64

modalities = {
    'tabular': torch.randn(batch_size, 10, channels),
    'imaging': torch.randn(batch_size, 100, channels),
    'timeseries': torch.randn(batch_size, 50, channels),
}

print(f"\nInput Modalities:")
for name, feat in modalities.items():
    print(f"  {name:12s}: {list(feat.shape)}")

print("\n" + "-"*70)

# Test 1: Cross-Modal Attention
print("\n1. Cross-Modal Attention Fusion")
fusion1 = CrossModalAttentionFusion(channels=channels, num_heads=4)
out1 = fusion1(modalities)
print(f"   Output shape: {list(out1.shape)}")
print(f"   Method: Multi-head attention between modalities")

# Test 2: Gated Fusion
print("\n2. Gated Fusion")
fusion2 = GatedFusion(channels=channels)
out2 = fusion2(modalities)
print(f"   Output shape: {list(out2.shape)}")
print(f"   Method: Learnable importance weights per modality")

# Test 3: Hierarchical Fusion
print("\n3. Hierarchical Fusion")
fusion3 = HierarchicalFusion(channels=channels, fusion_type='attention')
out3 = fusion3(modalities)
print(f"   Output shape: {list(out3.shape)}")
print(f"   Method: Intra-modality + inter-modality fusion")

print("\n" + "="*70)
print("✓ All advanced fusion strategies working!")
print("="*70)
