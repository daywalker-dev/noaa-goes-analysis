"""Variational Transformer: probabilistic temporal forecasting with ELBO."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import SinusoidalPE


class MeteoProjector(nn.Module):
    """MLP to project meteorological fields to model dimension."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VariationalTransformer(nn.Module):
    """Variational Transformer for probabilistic multi-step atmospheric forecasting.

    Architecture:
        Input projection → Sinusoidal PE → Transformer encoder → VAE bottleneck
        → Transformer decoder with learnable query tokens → μ + log σ² output heads

    Args:
        latent_dim: Input dimension of spatial latent embeddings.
        meteo_dim: Dimension of meteorological input features.
        state_dim: Dimension of the predicted atmospheric state vector.
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_encoder_layers: Number of transformer encoder layers.
        n_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: FFN inner dimension.
        dropout: Dropout rate.
        forecast_steps: Number of future steps to predict.
        beta_kl: Weight on KL divergence in ELBO.
        free_bits: Minimum per-dimension KL (prevents posterior collapse).
    """

    def __init__(
        self,
        latent_dim: int = 384,
        meteo_dim: int = 5,
        state_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        forecast_steps: int = 24,
        beta_kl: float = 1.0,
        free_bits: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.forecast_steps = forecast_steps
        self.beta_kl = beta_kl
        self.free_bits = free_bits

        # Input projections
        self.latent_proj = nn.Linear(latent_dim, d_model)
        self.meteo_proj = MeteoProjector(meteo_dim, d_model)
        self.input_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pe = SinusoidalPE(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # VAE bottleneck
        self.to_mu = nn.Linear(d_model, d_model)
        self.to_logvar = nn.Linear(d_model, d_model)

        # Learnable forecast query tokens
        self.forecast_queries = nn.Parameter(torch.randn(1, forecast_steps, d_model) * 0.02)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        # Output heads
        self.out_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )
        self.out_logvar = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, state_dim),
        )

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ * ε."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # MAP estimate at eval

    def _compute_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL(q(z|x) || p(z)) with free-bits regularization."""
        # Per-dimension KL
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        # Free-bits: clamp per-dim KL from below
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        return kl_per_dim.sum(dim=-1).mean()

    def forward(
        self,
        spatial_latents: torch.Tensor,
        meteo_fields: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass producing distributional forecast.

        Args:
            spatial_latents: (B, T_in, latent_dim) from frozen encoders.
            meteo_fields: (B, T_in, F) meteorological variables.

        Returns:
            Dict with 'mu', 'logvar' (B, T_out, state_dim), 'kl_loss'.
        """
        B = spatial_latents.shape[0]

        # Project and fuse inputs
        lat_proj = self.latent_proj(spatial_latents)     # (B, T, D)
        met_proj = self.meteo_proj(meteo_fields)         # (B, T, D)
        fused = self.input_fusion(torch.cat([lat_proj, met_proj], dim=-1))
        fused = self.pe(fused)

        # Encode
        memory = self.encoder(fused)  # (B, T_in, D)

        # VAE bottleneck on pooled memory
        pooled = memory.mean(dim=1)              # (B, D)
        z_mu = self.to_mu(pooled)                # (B, D)
        z_logvar = self.to_logvar(pooled)        # (B, D)
        z_logvar = torch.clamp(z_logvar, -10, 10)
        z = self._reparameterize(z_mu, z_logvar) # (B, D)

        # Inject z into memory (broadcast to all positions)
        memory = memory + z.unsqueeze(1)

        # Decode with forecast queries
        queries = self.forecast_queries.expand(B, -1, -1)
        queries = self.pe(queries)
        decoded = self.decoder(queries, memory)  # (B, T_out, D)

        # Output distributions
        mu = self.out_mu(decoded)                # (B, T_out, state_dim)
        logvar = self.out_logvar(decoded)        # (B, T_out, state_dim)
        logvar = torch.clamp(logvar, -10, 10)

        kl_loss = self._compute_kl(z_mu, z_logvar)

        return {
            "mu": mu,
            "logvar": logvar,
            "kl_loss": kl_loss,
            "z": z,
        }

    @torch.no_grad()
    def sample(
        self,
        spatial_latents: torch.Tensor,
        meteo_fields: torch.Tensor,
        n_samples: int = 50,
    ) -> dict[str, torch.Tensor]:
        """Monte Carlo sampling for uncertainty estimation.

        Returns:
            Dict with 'mean', 'std', 'p05', 'p95' — all (B, T_out, state_dim).
        """
        was_training = self.training
        self.train()  # Enable stochastic sampling

        samples = []
        for _ in range(n_samples):
            out = self.forward(spatial_latents, meteo_fields)
            # Sample from predicted Gaussian
            std = torch.exp(0.5 * out["logvar"])
            sample = out["mu"] + std * torch.randn_like(std)
            samples.append(sample)

        self.train(was_training)
        samples = torch.stack(samples, dim=0)  # (N, B, T, D)

        return {
            "mean": samples.mean(dim=0),
            "std": samples.std(dim=0),
            "p05": torch.quantile(samples, 0.05, dim=0),
            "p95": torch.quantile(samples, 0.95, dim=0),
            "samples": samples,
        }
