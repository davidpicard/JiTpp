"""Image normalization utilities shared across training and inference.

Convention
----------
- Raw images from the DataLoader are uint8 tensors in [0, 255] (PILToTensor output).
- The model operates on float tensors normalized to [-1, 1].
- Display / logging uses float tensors in [0, 1].

All conversions must go through these helpers so the convention cannot silently
diverge between training, sampling, and visualization.
"""
import torch


def normalize(x: torch.Tensor) -> torch.Tensor:
    """[0, 255] uint8 → [-1, 1] float32, in-place friendly."""
    return x.to(torch.float32).div_(255.0).mul_(2.0).sub_(1.0)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float → [0, 1] float for display/saving."""
    return (x + 1.0) / 2.0
