"""
Simple example demonstrating multi-modal fusion in MedTabX.
"""

import sys
from pathlib import Path
import torch

# Add medtabx to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("MedTabX Multi-Modal Fusion Example")
print("="*70)

# Import fusion modules
from medtabx.interaction.fusion_base import MultiModalFusion
from medtabx.interaction.pooling_fusion import MeanPoolingFusion, ConcatPoolingFusion
from medtabx.interaction.attention_fusion import AttentionPoolingFusion

# Simulate encoded features from different modalities
batch_size = 8
channels = 64

modalities = {
    'tabular': torch.randn(batch_size, 10, channels),      # 10 columns
    'imaging': torch.randn(batch_size, 100, channels),     # 100 patches
    'timeseries': torch.randn(batch_size, 50, channels),   # 50 time steps
    'text': torch.randn(batch_size, 20, channels),         # 20 sentences
}

print(f"\nInput Modalities:")
for name, feat in modalities.items():
    print(f"  {name:12s}: {list(feat.shape)}")

print("\n" + "-"*70)
print("Testing Fusion Strategies")
print("-"*70)

# Test 1: Mean Pooling Fusion
print("\n1. Mean Pooling Fusion")
fusion1 = MeanPoolingFusion(channels=channels)
out1 = fusion1(modalities)
print(f"   Output shape: {list(out1.shape)}")
print(f"   Method: Average all modalities")

# Test 2: Concatenation Fusion
print("\n2. Concatenation Fusion")
fusion2 = ConcatPoolingFusion(channels=channels)
out2 = fusion2(modalities)
print(f"   Output shape: {list(out2.shape)}")
print(f"   Method: Concatenate all modalities")

# Test 3: Attention Pooling Fusion
print("\n3. Attention Pooling Fusion")
fusion3 = AttentionPoolingFusion(channels=channels)
out3 = fusion3(modalities)
print(f"   Output shape: {list(out3.shape)}")
print(f"   Method: Attention pooling + concatenation")

print("\n" + "="*70)
print("✓ All fusion strategies working!")
print("="*70)
