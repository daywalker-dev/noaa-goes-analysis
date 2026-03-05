"""
Shared neural-network building blocks used across all model stages.

Includes:
    - ResidualBlock          — pre-activation residual convolution
    - ChannelAttention       — SE-style channel squeeze-excitation
    - SpatialAttention       — max+avg pooled spatial gate
    - CBAM                   — channel + spatial attention (Woo et al. 2018)
    - FiLM                   — feature-wise linear modulation conditioning
    - SinusoidalPE           — sinusoidal positional encoding for sequences
    - DownBlock              — strided downsample + N residual blocks + optional CBAM
    - UpBlock                — transposed upsample + N residual blocks + optional skip
    - SpatialSelfAttention   — multi-head self-attention over H×W
    - ssim_loss              — differentiable 1-SSIM loss
    - PhysicsAwareLoss       — Laplacian smoothness + temporal conservation
    - CombinedLoss           — weighted MSE + SSIM + physics
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
    """Pre-activation residual block.

    ``GroupNorm → GELU → Conv → GroupNorm → GELU → Dropout → Conv → + skip``
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


# ===========================================================================
# Channel & Spatial Attention  (CBAM components)
# ===========================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention.

    Applies global avg-pool + max-pool, feeds through a shared MLP,
    then combines with sigmoid to produce per-channel scaling.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        # Global average and max pool → (B, C)
        avg = x.mean(dim=(-2, -1))
        mx = x.amax(dim=(-2, -1))
        # Shared MLP on both, combine
        scale = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention gate using channel-wise avg+max pooling.

    Concatenates avg and max across channels, passes through a 7×7 conv,
    applies sigmoid to produce a per-pixel spatial mask.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        mx = x.amax(dim=1, keepdim=True)    # (B, 1, H, W)
        cat = torch.cat([avg, mx], dim=1)   # (B, 2, H, W)
        mask = torch.sigmoid(self.conv(cat))  # (B, 1, H, W)
        return x * mask


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ===========================================================================
# FiLM  (Feature-wise Linear Modulation)
# ===========================================================================

class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning spatial features.

    Given a conditioning vector ``cond`` and spatial features ``x``,
    produces ``gamma * x + beta`` where gamma and beta are predicted
    from ``cond``.

    Args:
        cond_dim: Dimension of the conditioning input vector.
        channels: Number of spatial feature channels.
    """

    def __init__(self, cond_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma_fc = nn.Linear(cond_dim, channels)
        self.beta_fc = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features.
            cond: (B, cond_dim) conditioning vector.
        Returns:
            Modulated features (B, C, H, W).
        """
        gamma = self.gamma_fc(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta_fc(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ===========================================================================
# Sinusoidal Positional Encoding
# ===========================================================================

class SinusoidalPE(nn.Module):
    """Additive sinusoidal positional encoding for sequence inputs.

    Supports inputs of shape ``(B, T, D)``.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2 + d_model % 2])
        pe = pe.unsqueeze(0)  # (1, max_len, D)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D) with positional encoding added."""
        return x + self.pe[:, : x.size(1), : x.size(2)]


# ===========================================================================
# DownBlock  (encoder building block)
# ===========================================================================

class DownBlock(nn.Module):
    """Strided downsample → N residual blocks → optional CBAM.

    Supports two call signatures:
        DownBlock(in_ch, out_ch)                            — simple (tests)
        DownBlock(in_ch, out_ch, n_res_blocks, use_cbam, dropout)  — full (encoder)

    When ``n_res_blocks`` is omitted it defaults to 1; ``use_cbam`` defaults
    to False so the simple signature produces a lightweight block compatible
    with the existing unit tests.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_res_blocks: int = 1,
        use_cbam: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        blocks: list[nn.Module] = [ResidualBlock(out_ch, dropout) for _ in range(n_res_blocks)]
        if use_cbam:
            blocks.append(CBAM(out_ch))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.down(x))


# ===========================================================================
# UpBlock  (decoder building block)
# ===========================================================================

class UpBlock(nn.Module):
    """Transposed-conv upsample → optional skip concat → N residual blocks.

    Supports two call signatures:
        UpBlock(in_ch, out_ch)                          — simple (tests, no skip)
        UpBlock(in_ch, out_ch, n_res_blocks, dropout)   — full (encoder decoder)

    When invoked with a ``skip`` tensor, the skip features are concatenated
    along the channel axis and projected back down.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_res_blocks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)
        # Projection for skip connections (created lazily if needed)
        self._skip_proj: Optional[nn.Module] = None
        self._out_ch = out_ch
        blocks: list[nn.Module] = [ResidualBlock(out_ch, dropout) for _ in range(n_res_blocks)]
        self.body = nn.Sequential(*blocks)

    def _get_skip_proj(self, skip_ch: int, device: torch.device) -> nn.Module:
        """Lazily create a 1×1 projection to merge skip features."""
        if self._skip_proj is None or next(self._skip_proj.parameters()).shape[1] != self._out_ch + skip_ch:
            self._skip_proj = nn.Conv2d(
                self._out_ch + skip_ch, self._out_ch, 1, bias=False
            ).to(device)
        return self._skip_proj

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Centre-crop skip to match x if sizes differ
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            if dh > 0 or dw > 0:
                skip = skip[
                    :, :,
                    dh // 2 : dh // 2 + x.shape[2],
                    dw // 2 : dw // 2 + x.shape[3],
                ]
            x = torch.cat([x, skip], dim=1)
            x = self._get_skip_proj(skip.shape[1], x.device)(x)
        return self.body(x)


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
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
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
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
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
    """Physics-informed regularisation.

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