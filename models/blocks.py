"""
Shared neural-network building blocks used across all model stages.

Includes residual convolution blocks, spatial self-attention, SSIM loss,
and physics-aware loss terms.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Residual Block
# ===========================================================================

class ResidualBlock(nn.Module):
    """Pre-activation residual block with optional channel projection.

    ``Conv → GroupNorm → GELU → Conv → GroupNorm → GELU → + skip``
    """

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, channels), channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownBlock(nn.Module):
    """Strided convolution down-sample + residual."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.res = ResidualBlock(out_ch, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Transposed-convolution up-sample + residual, with optional skip."""

    def __init__(self, in_ch: int, out_ch: int, skip_ch: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        self.proj = nn.Conv2d(out_ch + skip_ch, out_ch, 1) if skip_ch else nn.Identity()
        self.res = ResidualBlock(out_ch, dropout)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Centre-crop skip to match x if sizes differ
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh > 0 or dw > 0:
                skip = skip[:, :, dh // 2 : dh // 2 + x.shape[2],
                             dw // 2 : dw // 2 + x.shape[3]]
            x = torch.cat([x, skip], dim=1)
            x = self.proj(x)
        return self.res(x)


# ===========================================================================
# Spatial Self-Attention
# ===========================================================================

class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over spatial dims (H×W)."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)        # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


# ===========================================================================
# Loss Functions
# ===========================================================================

def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    reduction: str = "mean",
) -> torch.Tensor:
    """Differentiable structural similarity index loss (1 − SSIM).

    Operates on (B, C, H, W) tensors.
    """
    C = pred.shape[1]
    # Gaussian kernel
    sigma = 1.5
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = (g.unsqueeze(0) * g.unsqueeze(1))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    pad = window_size // 2
    mu_x = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(target, kernel, padding=pad, groups=C)

    mu_xx = mu_x * mu_x
    mu_yy = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu_xx
    sigma_yy = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu_yy
    sigma_xy = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu_xy

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
    )

    if reduction == "mean":
        return 1.0 - ssim_map.mean()
    return 1.0 - ssim_map


class PhysicsAwareLoss(nn.Module):
    """Optional physics-informed regularisation.

    Penalises violations of spatial smoothness (Laplacian) and
    conservation tendencies (temporal gradient magnitude).
    """

    def __init__(self, smoothness_weight: float = 1.0, conservation_weight: float = 1.0) -> None:
        super().__init__()
        self.sw = smoothness_weight
        self.cw = conservation_weight
        # Laplacian kernel
        lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("laplacian", lap.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = pred.shape
        loss = torch.tensor(0.0, device=pred.device)

        # Spatial smoothness: ||∇²(pred − target)||²
        diff = pred - target
        for c in range(C):
            lap_diff = F.conv2d(diff[:, c : c + 1], self.laplacian, padding=1)
            loss = loss + self.sw * (lap_diff ** 2).mean()

        # Temporal conservation: penalise large jumps from previous step
        if prev is not None:
            dt_pred = pred - prev
            dt_true = target - prev
            loss = loss + self.cw * F.mse_loss(dt_pred, dt_true)

        return loss


class CombinedLoss(nn.Module):
    """Weighted combination of MSE + SSIM + optional physics loss."""

    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.3,
        physics_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.mse_w = mse_weight
        self.ssim_w = ssim_weight
        self.phys_w = physics_weight
        self.physics = PhysicsAwareLoss() if physics_weight > 0 else None

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        prev: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            pred = pred * mask
            target = target * mask

        loss = self.mse_w * F.mse_loss(pred, target)
        if self.ssim_w > 0:
            loss = loss + self.ssim_w * ssim_loss(pred, target)
        if self.physics is not None and self.phys_w > 0:
            loss = loss + self.phys_w * self.physics(pred, target, prev)
        return loss