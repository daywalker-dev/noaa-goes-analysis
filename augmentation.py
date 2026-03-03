"""Spatial and temporal augmentation transforms for training."""
from __future__ import annotations

import numpy as np
import torch


class RandomFlip:
    """Random horizontal and/or vertical flip for (C, H, W) tensors."""

    def __init__(self, horizontal: bool = True, vertical: bool = True, p: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.horizontal and torch.rand(1).item() < self.p:
            x = torch.flip(x, dims=[-1])
        if self.vertical and torch.rand(1).item() < self.p:
            x = torch.flip(x, dims=[-2])
        return x


class TemporalJitter:
    """Small random offset to temporal window start (data-level augmentation)."""

    def __init__(self, max_offset: int = 1):
        self.max_offset = max_offset

    def get_offset(self) -> int:
        return np.random.randint(-self.max_offset, self.max_offset + 1)


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


def build_augmentation(cfg) -> Compose | None:
    """Build augmentation pipeline from config."""
    aug_cfg = cfg.data.get("augmentation", {})
    transforms = []
    if aug_cfg.get("horizontal_flip", False) or aug_cfg.get("vertical_flip", False):
        transforms.append(RandomFlip(
            horizontal=aug_cfg.get("horizontal_flip", False),
            vertical=aug_cfg.get("vertical_flip", False),
        ))
    return Compose(transforms) if transforms else None
