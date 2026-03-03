"""Fusion Transformer: combines all prediction streams into unified forecast."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionTransformer(nn.Module):
    """Fuses CNN latents, Bayesian forecasts, meteo fields, and generator
    reconstructions into a single coherent atmospheric forecast with uncertainty.

    Architecture:
        Source projectors → Source type embeddings → Transformer encoder
        → Mean-aggregate across sources → Forecast + Uncertainty heads

    Args:
        cnn_dim: Dimension of CNN latent embeddings.
        bayes_dim: Dimension of Bayesian output (mu + logvar → 2 * state_dim).
        meteo_dim: Dimension of meteorological features.
        gen_dim: Dimension of flattened generator features.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        dim_feedforward: FFN inner dimension.
        dropout: Dropout rate.
        n_sources: Number of input sources (default 4).
        out_channels: Number of output L2 channels.
        spatial_size: (H, W) output spatial size.
    """

    def __init__(
        self,
        cnn_dim: int = 384,
        bayes_dim: int = 512,
        meteo_dim: int = 5,
        gen_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_sources: int = 4,
        out_channels: int = 20,
        spatial_size: tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sources = n_sources
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        # Source projectors
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.bayes_proj = nn.Sequential(
            nn.Linear(bayes_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.meteo_proj = nn.Sequential(
            nn.Linear(meteo_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )
        self.gen_proj = nn.Sequential(
            nn.Linear(gen_dim, d_model), nn.LayerNorm(d_model), nn.GELU()
        )

        # Source-type embeddings
        self.source_embeddings = nn.Embedding(n_sources, d_model)

        # Transformer encoder for fusion
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Forecast head: produces per-channel spatial scales
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, out_channels),
        )

        # Uncertainty head: produces uncertainty modulated by Bayesian variance
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
            nn.Softplus(),
        )

    def forward(
        self,
        cnn_latents: torch.Tensor,
        bayes_output: torch.Tensor,
        meteo_fields: torch.Tensor,
        gen_features: torch.Tensor,
        gen_spatial: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse all sources into a unified forecast.

        Args:
            cnn_latents: (B, T, cnn_dim) CNN encoder latents.
            bayes_output: (B, T, bayes_dim) Bayesian μ and σ² concatenated.
            meteo_fields: (B, T, meteo_dim) meteorological fields.
            gen_features: (B, T, gen_dim) generator feature summary.
            gen_spatial: (B, T, C, H, W) optional generator spatial output.

        Returns:
            Dict with 'forecast' (B, T, C, H, W) and 'uncertainty' (B, T, C).
        """
        B, T, _ = cnn_latents.shape

        # Project each source
        src_cnn = self.cnn_proj(cnn_latents)       # (B, T, D)
        src_bayes = self.bayes_proj(bayes_output)   # (B, T, D)
        src_meteo = self.meteo_proj(meteo_fields)   # (B, T, D)
        src_gen = self.gen_proj(gen_features)        # (B, T, D)

        # Add source-type embeddings
        src_ids = torch.arange(self.n_sources, device=cnn_latents.device)
        src_embs = self.source_embeddings(src_ids)  # (4, D)

        src_cnn = src_cnn + src_embs[0]
        src_bayes = src_bayes + src_embs[1]
        src_meteo = src_meteo + src_embs[2]
        src_gen = src_gen + src_embs[3]

        # Interleave: (B, T*4, D)
        tokens = torch.stack([src_cnn, src_bayes, src_meteo, src_gen], dim=2)
        tokens = tokens.reshape(B, T * self.n_sources, self.d_model)

        # Transformer fusion
        fused = self.transformer(tokens)  # (B, T*4, D)

        # Mean-aggregate back to (B, T, D)
        fused = fused.reshape(B, T, self.n_sources, self.d_model).mean(dim=2)

        # Forecast: channel-wise scaling applied to generator output
        scales = self.forecast_head(fused)       # (B, T, C)
        uncertainty = self.uncertainty_head(fused)  # (B, T, C)

        result = {
            "scales": scales,
            "uncertainty": uncertainty,
            "fused_features": fused,
        }

        # If generator spatial output available, apply scales
        if gen_spatial is not None:
            # scales: (B, T, C) → (B, T, C, 1, 1) for broadcasting
            forecast = gen_spatial * scales.unsqueeze(-1).unsqueeze(-1)
            result["forecast"] = forecast

        return result
