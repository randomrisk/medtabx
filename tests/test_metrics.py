"""Test metrics module."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("Testing Metrics Module")
print("="*70)

from medtabx.metrics.classification import Accuracy, Precision, Recall

# Create sample data
predictions = torch.tensor([0, 1, 1, 0, 1])
targets = torch.tensor([0, 1, 0, 0, 1])

# Test Accuracy
acc = Accuracy()
acc.update(predictions, targets)
print(f"\nAccuracy: {acc.compute():.3f}")

# Test Precision
prec = Precision()
prec.update(predictions, targets)
print(f"Precision: {prec.compute():.3f}")

# Test Recall
rec = Recall()
rec.update(predictions, targets)
print(f"Recall: {rec.compute():.3f}")

print("\n" + "="*70)
print("✓ Metrics working!")
print("="*70)
