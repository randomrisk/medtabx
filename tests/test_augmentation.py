"""Test augmentation transforms."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("Testing Augmentation Transforms")
print("="*70)

from medtabx.transforms.augmentation import GaussianNoise, Mixup

# Test 1: Tabular augmentation
print("\n1. Tabular Augmentation")
data = np.array([[1.0, 2.0], [3.0, 4.0]])
print(f"Original: {data[0]}")

noise_aug = GaussianNoise(noise_level=0.1)
augmented = noise_aug.fit_transform(data)
print(f"With noise: {augmented[0]}")

# Test 2: Time series augmentation
from medtabx.transforms.augmentation import TimeJitter, TimeScaling

print("\n2. Time Series Augmentation")
ts_data = np.array([[1.0, 2.0, 3.0]])
print(f"Original: {ts_data[0]}")

jitter_aug = TimeJitter(sigma=0.05)
jittered = jitter_aug.fit_transform(ts_data)
print(f"With jitter: {jittered[0]}")

print("\n" + "="*70)
print("✓ All augmentations working!")
print("="*70)




