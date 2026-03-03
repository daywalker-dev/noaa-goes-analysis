"""Deterministic seeding and reproducibility utilities."""
from __future__ import annotations

import os
import random
import sys
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seeds for random, numpy, torch, and CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_deterministic(enabled: bool = True) -> None:
    """Enable/disable deterministic algorithms (may reduce performance)."""
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(enabled)
        except RuntimeError:
            pass  # some ops don't have deterministic implementations


def get_environment_info() -> dict[str, str]:
    """Capture environment info for reproducibility logs."""
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "N/A"
        info["gpu_count"] = str(torch.cuda.device_count())
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
