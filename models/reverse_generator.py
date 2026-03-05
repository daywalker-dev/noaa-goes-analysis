"""Conditional UNet: maps predicted atmospheric state → synthetic GOES L2 fields."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ResidualBlock, FiLM


class ConditionalUNet(nn.Module):
    """Conditional UNet generator mapping atmospheric state vectors to spatial fields.

    Uses a learnable spatial prior, FiLM conditioning at each decoder level,
    and optional spectral normalization on the output head.

    Args:
        state_dim: Dimension of the conditioning state vector.
        out_channels: Number of output L2 channels.
        noise_dim: Dimension of optional noise input for diversity.
        base_channels: Base channel width.
        channel_multipliers: Channel multipliers per stage (from deepest to shallowest).
        n_res_blocks: Residual blocks per decoder stage.
        use_spectral_norm: Whether to apply spectral norm on output.
        initial_spatial: (H, W) of the learned spatial prior.
    """

    def __init__(
        self,
        state_dim: int = 256,
        out_channels: int = 20,
        noise_dim: int = 32,
        base_channels: int = 64,
        channel_multipliers: list[int] = (8, 4, 2, 1),
        n_res_blocks: int = 2,
        use_spectral_norm: bool = True,
        initial_spatial: tuple[int, int] = (8, 8),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.out_channels = out_channels

        channels = [base_channels * m for m in channel_multipliers]
        cond_dim = state_dim + noise_dim

        # Learnable spatial prior
        self.spatial_prior = nn.Parameter(
            torch.randn(1, channels[0], initial_spatial[0], initial_spatial[1]) * 0.02
        )

        # Condition embedding MLP
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.GELU(),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim),
        )

        # FiLM layers (one per decoder stage)
        self.film_layers = nn.ModuleList()
        for ch in channels:
            self.film_layers.append(FiLM(cond_dim, ch))

        # Decoder stages
        self.decoder_stages = nn.ModuleList()
        for i in range(len(channels) - 1):
            stage = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=False),
                nn.GroupNorm(min(8, channels[i + 1]), channels[i + 1]),
                nn.GELU(),
                *[ResidualBlock(channels[i + 1]) for _ in range(n_res_blocks)],
            )
            self.decoder_stages.append(stage)

        # Output head
        out_conv = nn.Conv2d(channels[-1], out_channels, 1)
        if use_spectral_norm:
            out_conv = nn.utils.spectral_norm(out_conv)
        self.output_head = out_conv

    def forward(
        self,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        target_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Generate synthetic L2 fields from atmospheric state.

        Args:
            state: (B, state_dim) predicted atmospheric state.
            noise: (B, noise_dim) optional noise for diversity. If None, zeros used.
            target_size: (H, W) desired output spatial size.

        Returns:
            Synthetic L2 fields (B, out_channels, H, W).
        """
        B = state.shape[0]

        # Build conditioning vector
        if noise is None:
            noise = torch.zeros(B, self.noise_dim, device=state.device)
        cond = self.cond_embed(torch.cat([state, noise], dim=-1))  # (B, cond_dim)

        # Start from learned spatial prior
        h = self.spatial_prior.expand(B, -1, -1, -1)  # (B, C0, h0, w0)

        # Apply FiLM at first stage
        h = self.film_layers[0](h, cond)

        # Decoder stages with FiLM conditioning
        for i, stage in enumerate(self.decoder_stages):
            h = stage(h)
            h = self.film_layers[i + 1](h, cond)

        # Output
        out = self.output_head(h)

        if target_size is not None and out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)

        return out

    def decode_sequence(
        self,
        states: torch.Tensor,
        target_size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Decode a sequence of state vectors.

        Args:
            states: (B, T, state_dim) sequence of atmospheric states.
            target_size: (H, W) for output.

        Returns:
            (B, T, out_channels, H, W) synthetic L2 fields.
        """
        B, T, D = states.shape
        # Flatten batch and time
        flat_states = states.reshape(B * T, D)
        flat_out = self.forward(flat_states, target_size=target_size)
        C, H, W = flat_out.shape[1:]
        return flat_out.reshape(B, T, C, H, W)
