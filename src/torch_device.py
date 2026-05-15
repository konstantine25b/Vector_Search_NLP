"""Shared device selection for PyTorch scripts (CUDA / Apple MPS / CPU)."""

from __future__ import annotations

import torch


def pick_torch_device(name: str = "auto") -> torch.device:
    """
    auto: CUDA if available, else Apple Metal (mps), else CPU.
    """
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
