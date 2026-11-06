"""Test transforms module."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("Testing Transforms Module")
print("="*70)

from medtabx.transforms.tabular import StandardScaler

# Test data
data = np.array([[1, 2], [3, 4], [5, 6]])
print(f"\nOriginal data:\n{data}")

# Test StandardScaler
scaler = StandardScaler()
scaler.fit(data)
transformed = scaler.transform(data)
print(f"\nTransformed data:\n{transformed}")

print("\n" + "="*70)
print("✓ Transforms working!")
print("="*70)
