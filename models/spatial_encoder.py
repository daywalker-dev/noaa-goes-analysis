"""Domain-specific spatial CNN encoders with CBAM attention."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from goes_forecast.models.blocks import DownBlock, UpBlock, ResidualBlock, CBAM


class SpatialCNNEncoder(nn.Module):
    """Spatial CNN encoder-decoder for a single domain.

    Architecture:
        Conv stem → DownBlocks with CBAM → Global avg pool → Latent FC
        Latent → spatial projection → UpBlocks with skip connections → Reconstruction

    Args:
        in_channels: Number of input channels for this domain.
        latent_dim: Dimension of the latent embedding vector.
        base_channels: Base channel width (multiplied at each stage).
        channel_multipliers: List of multipliers for each downsampling stage.
        n_res_blocks: Number of residual blocks per stage.
        use_cbam: Whether to use CBAM attention in down blocks.
        dropout: Dropout rate in residual blocks.
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int = 128,
        base_channels: int = 64,
        channel_multipliers: list[int] = (1, 2, 4, 8),
        n_res_blocks: int = 4,
        use_cbam: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.n_stages = len(channel_multipliers)

        # Channel widths at each stage
        channels = [base_channels * m for m in channel_multipliers]

        # --- Encoder ---
        g = min(8, base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=1, padding=3, bias=False),
            nn.GroupNorm(g, base_channels),
            nn.GELU(),
        )

        self.encoder_stages = nn.ModuleList()
        prev_ch = base_channels
        for ch in channels:
            self.encoder_stages.append(
                DownBlock(prev_ch, ch, n_res_blocks, use_cbam, dropout)
            )
            prev_ch = ch

        # Latent projection
        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # --- Decoder ---
        # Project latent back to spatial feature map
        self.bottleneck_spatial = 8  # spatial size at bottleneck
        self.latent_to_spatial = nn.Sequential(
            nn.Linear(latent_dim, channels[-1] * self.bottleneck_spatial ** 2),
            nn.GELU(),
        )

        self.decoder_stages = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):
            self.decoder_stages.append(
                UpBlock(rev_channels[i], rev_channels[i + 1], n_res_blocks, dropout)
            )
        # Final upblock back to base_channels
        self.decoder_stages.append(
            UpBlock(rev_channels[-1], base_channels, n_res_blocks, dropout)
        )

        # Output head
        self.output_head = nn.Conv2d(base_channels, in_channels, 1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode input to latent vector, returning skip connections.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            latent: (B, latent_dim)
            skips: List of intermediate feature maps for decoder.
        """
        h = self.stem(x)
        skips = [h]
        for stage in self.encoder_stages:
            h = stage(h)
            skips.append(h)
        latent = self.to_latent(h)
        return latent, skips

    def decode(self, latent: torch.Tensor, skips: list[torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            latent: (B, latent_dim)
            skips: Encoder skip connections.
            target_size: (H, W) of desired output.

        Returns:
            Reconstruction (B, C, H, W).
        """
        B = latent.shape[0]
        ch = skips[-1].shape[1]
        h = self.latent_to_spatial(latent).view(B, ch, self.bottleneck_spatial, self.bottleneck_spatial)

        # Resize to match deepest skip
        if h.shape[-2:] != skips[-1].shape[-2:]:
            h = F.interpolate(h, size=skips[-1].shape[-2:], mode="bilinear", align_corners=False)

        # Decoder with skip connections (reversed order, skip stem features)
        rev_skips = list(reversed(skips[:-1]))  # exclude the deepest (already used)
        for i, up_block in enumerate(self.decoder_stages):
            skip = rev_skips[i] if i < len(rev_skips) else None
            h = up_block(h, skip)

        # Final resize to exact target
        if h.shape[-2:] != target_size:
            h = F.interpolate(h, size=target_size, mode="bilinear", align_corners=False)

        return self.output_head(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass: encode → decode.

        Returns dict with 'latent' and 'reconstruction'.
        """
        latent, skips = self.encode(x)
        recon = self.decode(latent, skips, x.shape[-2:])
        return {"latent": latent, "reconstruction": recon}

    def encode_only(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without decoder pass (for frozen inference)."""
        latent, _ = self.encode(x)
        return latent


class DomainEncoderEnsemble(nn.Module):
    """Ensemble of domain-specific SpatialCNNEncoders.

    Manages land, sea, and cloud encoders, routes inputs by domain,
    and concatenates latent embeddings.

    Args:
        domain_channels: Dict mapping domain name → number of input channels.
        encoder_cfg: Encoder config (latent_dim, n_res_blocks, etc.).
    """

    def __init__(self, domain_channels: dict[str, int], encoder_cfg):
        super().__init__()
        self.domain_names = sorted(domain_channels.keys())
        self.encoders = nn.ModuleDict()

        for domain in self.domain_names:
            self.encoders[domain] = SpatialCNNEncoder(
                in_channels=domain_channels[domain],
                latent_dim=encoder_cfg.latent_dim,
                base_channels=encoder_cfg.base_channels,
                channel_multipliers=list(encoder_cfg.channel_multipliers),
                n_res_blocks=encoder_cfg.n_res_blocks,
                use_cbam=encoder_cfg.use_cbam,
                dropout=encoder_cfg.dropout,
            )

        self.combined_latent_dim = encoder_cfg.latent_dim * len(self.domain_names)

    def forward(
        self,
        domain_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run all domain encoders.

        Args:
            domain_inputs: {domain_name: (B, C_domain, H, W)}

        Returns:
            Dict with 'latents' (B, combined_dim), per-domain 'recon_{domain}',
            and per-domain 'latent_{domain}'.
        """
        all_latents = []
        results = {}

        for domain in self.domain_names:
            if domain not in domain_inputs:
                continue
            out = self.encoders[domain](domain_inputs[domain])
            all_latents.append(out["latent"])
            results[f"latent_{domain}"] = out["latent"]
            results[f"recon_{domain}"] = out["reconstruction"]

        results["latents"] = torch.cat(all_latents, dim=-1)
        return results

    def encode_only(self, domain_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode all domains and concatenate latents (no decoding)."""
        latents = []
        for domain in self.domain_names:
            if domain in domain_inputs:
                latents.append(self.encoders[domain].encode_only(domain_inputs[domain]))
        return torch.cat(latents, dim=-1)

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        for p in self.parameters():
            p.requires_grad = True
