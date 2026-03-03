"""Reusable neural network building blocks for all model stages."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Residual Block (pre-activation: GroupNorm → GELU → Conv)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Pre-activation residual block with GroupNorm."""

    def __init__(self, channels: int, dropout: float = 0.0, groups: int = 8):
        super().__init__()
        g = min(groups, channels)
        self.block = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(g, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# ---------------------------------------------------------------------------
# Channel Attention (squeeze-excitation with avg + max pooling)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_pool = x.mean(dim=(2, 3))              # (B, C)
        max_pool = x.amax(dim=(2, 3))              # (B, C)
        attn = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Spatial Attention (channel-pooled conv gate)
# ---------------------------------------------------------------------------
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


# ---------------------------------------------------------------------------
# CBAM: Channel + Spatial Attention
# ---------------------------------------------------------------------------
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ---------------------------------------------------------------------------
# FiLM (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------
class FiLM(nn.Module):
    """Feature-wise Linear Modulation: γ * x + β from conditioning vector."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map (B, C, H, W)
            cond: Conditioning vector (B, cond_dim)
        """
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


# ---------------------------------------------------------------------------
# Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------
class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add PE to input. x shape: (B, T, D)."""
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Down/Up Blocks for encoder/decoder architectures
# ---------------------------------------------------------------------------
class DownBlock(nn.Module):
    """Downsample block: Conv2d(stride=2) → ResBlocks → optional CBAM."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_res_blocks: int = 2,
        use_cbam: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        blocks = [ResidualBlock(out_ch, dropout) for _ in range(n_res_blocks)]
        if use_cbam:
            blocks.append(CBAM(out_ch))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class UpBlock(nn.Module):
    """Upsample block: Bilinear up → Conv → ResBlocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        g = min(8, out_ch)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_ch, dropout) for _ in range(n_res_blocks)]
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip  # additive skip connection
        x = self.conv(x)
        return self.blocks(x)
