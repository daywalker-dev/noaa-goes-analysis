"""
Temporal Probabilistic Model
=============================
Combines latent embeddings from spatial encoders with auxiliary atmospheric
drivers (wind, humidity, temperature, pressure) and produces multi-step
probabilistic forecasts with mean + variance.

Three variant back-ends are supported:
    1. Variational RNN  (default)
    2. Bayesian LSTM
    3. Transformer with uncertainty head
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ===========================================================================
# Variational RNN
# ===========================================================================

class VariationalRNN(nn.Module):
    """
    Latent-variable sequential model.

    At each time step *t*:
        1.  Encode the input ``x_t`` and the previous hidden state into
            a posterior distribution ``q(z_t | x_≤t)``.
        2.  Sample ``z_t`` via the reparametrisation trick.
        3.  Decode ``z_t`` to produce mean and log-variance of the
            predicted atmospheric state at *t+1*.

    Parameters
    ----------
    input_dim : int
        Dimension of the concatenated input vector (latents + aux features).
    hidden_dim : int
        RNN hidden-state size.
    latent_dim : int
        Stochastic latent variable size.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout between GRU layers.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Recurrent backbone
        self.gru = nn.GRU(
            hidden_dim + latent_dim,  # concat z from prev step
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Prior  p(z_t | h_{t-1})
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu, log_var
        )

        # Posterior  q(z_t | h_{t-1}, x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        # Decoder  p(x_{t+1} | z_t, h_t)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim * 2),  # mean, log_var of output
        )

    # ----- helpers ----
    @staticmethod
    def _reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _split_params(h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def _kl_divergence(
        q_mu: torch.Tensor, q_logvar: torch.Tensor,
        p_mu: torch.Tensor, p_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL(q || p) for diagonal Gaussians."""
        kl = 0.5 * (
            p_logvar - q_logvar
            + (q_logvar.exp() + (q_mu - p_mu) ** 2) / p_logvar.exp()
            - 1.0
        )
        return kl.sum(dim=-1).mean()

    # ----- forward ----
    def forward(
        self,
        x_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x_seq : (B, T, input_dim)
            Sequence of concatenated latent + aux vectors.
        hidden : (num_layers, B, hidden_dim), optional

        Returns
        -------
        dict with keys:
            pred_mu    : (B, T, input_dim)
            pred_logvar: (B, T, input_dim)
            kl_loss    : scalar
            hidden     : (num_layers, B, hidden_dim)
        """
        B, T, D = x_seq.shape
        device = x_seq.device

        if hidden is None:
            hidden = torch.zeros(self.num_layers, B, self.hidden_dim, device=device)

        x_proj = self.input_proj(x_seq)  # (B, T, hidden_dim)

        z_t = torch.zeros(B, self.latent_dim, device=device)
        kl_total = torch.tensor(0.0, device=device)

        pred_mus, pred_logvars = [], []

        for t in range(T):
            h_prev = hidden[-1]  # last layer hidden

            # --- prior ---
            prior_params = self.prior_net(h_prev)
            p_mu, p_logvar = self._split_params(prior_params)

            # --- posterior (uses current input) ---
            post_input = torch.cat([h_prev, x_proj[:, t]], dim=-1)
            post_params = self.posterior_net(post_input)
            q_mu, q_logvar = self._split_params(post_params)

            # --- sample z ---
            if self.training:
                z_t = self._reparameterise(q_mu, q_logvar)
            else:
                z_t = q_mu  # deterministic at eval

            kl_total = kl_total + self._kl_divergence(q_mu, q_logvar, p_mu, p_logvar)

            # --- step GRU ---
            gru_in = torch.cat([x_proj[:, t], z_t], dim=-1).unsqueeze(1)
            _, hidden = self.gru(gru_in, hidden)

            # --- decode prediction ---
            dec_in = torch.cat([hidden[-1], z_t], dim=-1)
            out_params = self.decoder(dec_in)
            o_mu, o_logvar = self._split_params(out_params)
            pred_mus.append(o_mu)
            pred_logvars.append(o_logvar)

        return {
            "pred_mu": torch.stack(pred_mus, dim=1),
            "pred_logvar": torch.stack(pred_logvars, dim=1),
            "kl_loss": kl_total / T,
            "hidden": hidden,
        }

    def forecast(
        self,
        x_last: torch.Tensor,
        hidden: torch.Tensor,
        steps: int = 4,
        num_samples: int = 50,
    ) -> Dict[str, torch.Tensor]:
        """Autoregressive multi-step forecast with uncertainty.

        Returns
        -------
        dict with keys:
            mean   : (B, steps, input_dim)
            std    : (B, steps, input_dim)
            samples: (num_samples, B, steps, input_dim)
        """
        B, D = x_last.shape
        device = x_last.device

        all_samples = []
        for _ in range(num_samples):
            h = hidden.clone()
            x_t = x_last.clone()
            sample_seq = []

            for _ in range(steps):
                x_proj = self.input_proj(x_t).unsqueeze(1)
                h_prev = h[-1]

                prior_params = self.prior_net(h_prev)
                p_mu, p_logvar = self._split_params(prior_params)
                z_t = self._reparameterise(p_mu, p_logvar)

                gru_in = torch.cat([x_proj[:, 0], z_t], dim=-1).unsqueeze(1)
                _, h = self.gru(gru_in, h)

                dec_in = torch.cat([h[-1], z_t], dim=-1)
                out_params = self.decoder(dec_in)
                o_mu, _ = self._split_params(out_params)
                sample_seq.append(o_mu)
                x_t = o_mu  # autoregressive

            all_samples.append(torch.stack(sample_seq, dim=1))

        samples = torch.stack(all_samples, dim=0)  # (S, B, steps, D)
        return {
            "mean": samples.mean(dim=0),
            "std": samples.std(dim=0),
            "samples": samples,
        }


# ===========================================================================
# Bayesian LSTM variant
# ===========================================================================

class BayesianLSTM(nn.Module):
    """Simplified Bayesian LSTM using MC-Dropout for uncertainty.

    Keeps dropout *on* at inference to produce a distribution of outputs
    via multiple forward passes.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mc_drop = nn.Dropout(dropout)

        self.output_mu = nn.Linear(hidden_dim, input_dim)
        self.output_logvar = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        h = self.input_proj(x_seq)
        h, hidden_out = self.lstm(h, hidden)
        h = self.mc_drop(h)  # always active for Bayesian inference
        return {
            "pred_mu": self.output_mu(h),
            "pred_logvar": self.output_logvar(h),
            "kl_loss": torch.tensor(0.0, device=x_seq.device),
            "hidden": hidden_out,
        }


# ===========================================================================
# Transformer with Uncertainty Head
# ===========================================================================

class UncertaintyTransformer(nn.Module):
    """Causal Transformer encoder with Gaussian output heads."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (sinusoidal)
        self.pos_dim = hidden_dim
        self.register_buffer("_pos_cache", torch.zeros(1))  # placeholder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_mu = nn.Linear(hidden_dim, input_dim)
        self.output_logvar = nn.Linear(hidden_dim, input_dim)

    def _sinusoidal_pe(self, T: int, D: int, device: torch.device) -> torch.Tensor:
        pe = torch.zeros(T, D, device=device)
        pos = torch.arange(T, device=device, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: D // 2])  # handle odd D
        return pe.unsqueeze(0)  # (1, T, D)

    def forward(
        self,
        x_seq: torch.Tensor,
        hidden: Optional[Any] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x_seq.shape
        h = self.input_proj(x_seq) + self._sinusoidal_pe(T, self.input_proj.out_features, x_seq.device)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x_seq.device)
        h = self.transformer(h, mask=mask, is_causal=True)

        return {
            "pred_mu": self.output_mu(h),
            "pred_logvar": self.output_logvar(h),
            "kl_loss": torch.tensor(0.0, device=x_seq.device),
            "hidden": None,
        }


# ===========================================================================
# Factory
# ===========================================================================

_TEMPORAL_REGISTRY = {
    "variational_rnn": VariationalRNN,
    "bayesian_lstm": BayesianLSTM,
    "transformer": UncertaintyTransformer,
}


def build_temporal_model(cfg_section: Dict[str, Any]) -> nn.Module:
    """Construct a temporal model from the ``temporal`` config block."""
    model_type = cfg_section.get("type", "variational_rnn")
    cls = _TEMPORAL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(f"Unknown temporal model type: {model_type}. "
                         f"Choose from {list(_TEMPORAL_REGISTRY)}")
    return cls(
        input_dim=cfg_section["input_dim"],
        hidden_dim=cfg_section["hidden_dim"],
        latent_dim=cfg_section["latent_dim"],
        num_layers=cfg_section.get("num_layers", 2),
        dropout=cfg_section.get("dropout", 0.15),
        **({} if model_type != "transformer" else {"num_heads": cfg_section.get("num_heads", 8)}),
    )