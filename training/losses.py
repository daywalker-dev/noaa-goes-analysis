"""Loss functions for all training stages."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Masked MSE
# ---------------------------------------------------------------------------
class MaskedMSE(nn.Module):
    """MSE loss applied only to valid (non-masked) pixels."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (pred - target) ** 2
        if mask is not None:
            diff = diff * mask
            return diff.sum() / (mask.sum().clamp(min=1))
        return diff.mean()


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------
class SSIMLoss(nn.Module):
    """1 - SSIM loss. Uses pytorch-msssim if available, else simple approximation."""

    def __init__(self, data_range: float = 1.0, win_size: int = 7):
        super().__init__()
        self.data_range = data_range
        self.win_size = win_size
        self._has_msssim = False
        try:
            from pytorch_msssim import ssim
            self._ssim_fn = ssim
            self._has_msssim = True
        except ImportError:
            pass

    def _simple_ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Simple SSIM approximation using average pooling."""
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        k = self.win_size
        pad = k // 2

        mu_x = F.avg_pool2d(x, k, stride=1, padding=pad)
        mu_y = F.avg_pool2d(y, k, stride=1, padding=pad)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = F.avg_pool2d(x ** 2, k, stride=1, padding=pad) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(y ** 2, k, stride=1, padding=pad) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, k, stride=1, padding=pad) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self._has_msssim and pred.shape[-1] >= self.win_size and pred.shape[-2] >= self.win_size:
            return 1.0 - self._ssim_fn(pred, target, data_range=self.data_range, size_average=True)
        return 1.0 - self._simple_ssim(pred, target)


# ---------------------------------------------------------------------------
# CRPS Loss (Gaussian closed-form)
# ---------------------------------------------------------------------------
class CRPSLoss(nn.Module):
    """Closed-form CRPS for Gaussian distributions.

    CRPS(N(μ, σ²), y) = σ[ỹ·(2Φ(ỹ) - 1) + 2φ(ỹ) - 1/√π]
    where ỹ = (y - μ)/σ
    """

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        sigma = torch.exp(0.5 * logvar).clamp(min=1e-6)
        y_norm = (target - mu) / sigma

        # Standard normal PDF and CDF
        phi = torch.exp(-0.5 * y_norm ** 2) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + torch.erf(y_norm / math.sqrt(2)))

        crps = sigma * (y_norm * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))
        return crps.mean()


# ---------------------------------------------------------------------------
# Spectral Loss (2D FFT)
# ---------------------------------------------------------------------------
class SpectralLoss(nn.Module):
    """L2 distance between 2D FFT magnitude spectra."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, C, H, W)
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return F.mse_loss(pred_fft.abs(), target_fft.abs())


# ---------------------------------------------------------------------------
# Physics Constraint Loss
# ---------------------------------------------------------------------------
class PhysicsConstraintLoss(nn.Module):
    """Physics-informed regularization loss.

    Components:
        1. Temporal smoothness: penalize large jumps between consecutive steps.
        2. Spatial gradient consistency: penalize unrealistic spatial gradients.
        3. Energy conservation proxy: penalize large changes in spatial mean.
    """

    def __init__(
        self,
        temporal_weight: float = 1.0,
        spatial_weight: float = 0.5,
        energy_weight: float = 0.3,
    ):
        super().__init__()
        self.w_temporal = temporal_weight
        self.w_spatial = spatial_weight
        self.w_energy = energy_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            predictions: (B, T, C, H, W) or (B, C, H, W) predicted fields.
            targets: Optional ground truth for gradient comparison.
        """
        loss = torch.tensor(0.0, device=predictions.device)

        if predictions.dim() == 5:
            B, T, C, H, W = predictions.shape
            # Temporal smoothness
            if T > 1:
                diffs = predictions[:, 1:] - predictions[:, :-1]
                loss = loss + self.w_temporal * (diffs ** 2).mean()

            # Spatial gradients on last step
            pred_2d = predictions[:, -1]
        else:
            pred_2d = predictions

        # Spatial gradient consistency
        dx = pred_2d[:, :, :, 1:] - pred_2d[:, :, :, :-1]
        dy = pred_2d[:, :, 1:, :] - pred_2d[:, :, :-1, :]
        loss = loss + self.w_spatial * (dx.abs().mean() + dy.abs().mean())

        # Energy conservation proxy
        if targets is not None:
            tgt_2d = targets[:, -1] if targets.dim() == 5 else targets
            pred_mean = pred_2d.mean(dim=(-2, -1))
            tgt_mean = tgt_2d.mean(dim=(-2, -1))
            loss = loss + self.w_energy * F.mse_loss(pred_mean, tgt_mean)

        return loss


# ---------------------------------------------------------------------------
# ELBO Loss
# ---------------------------------------------------------------------------
class ELBOLoss(nn.Module):
    """Evidence lower bound loss: Reconstruction + β·KL.

    Supports KL annealing via set_beta().
    """

    def __init__(self, beta: float = 1.0, free_bits: float = 0.5):
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        self.recon_loss = MaskedMSE()

    def set_beta(self, beta: float) -> None:
        self.beta = beta

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        recon = self.recon_loss(mu, target, mask)

        # KL divergence with free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)
        kl = kl_per_dim.sum(dim=-1).mean()

        total = recon + self.beta * kl
        return {"total": total, "recon": recon, "kl": kl}


# ---------------------------------------------------------------------------
# Composite Loss (weighted sum builder)
# ---------------------------------------------------------------------------
class CompositeLoss(nn.Module):
    """Weighted sum of named loss components, driven by config."""

    def __init__(self, components: dict[str, nn.Module], weights: dict[str, float]):
        super().__init__()
        self.components = nn.ModuleDict(components)
        self.weights = weights

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Compute weighted sum. Each component receives relevant kwargs."""
        total = torch.tensor(0.0, device=next(iter(kwargs.values())).device
                             if kwargs else "cpu")
        losses = {}

        for name, module in self.components.items():
            w = self.weights.get(name, 1.0)
            if w <= 0:
                continue
            # Pass appropriate kwargs to each loss
            loss_val = module(**{k: v for k, v in kwargs.items()})
            if isinstance(loss_val, dict):
                losses.update({f"{name}/{k}": v for k, v in loss_val.items()})
                loss_val = loss_val.get("total", sum(loss_val.values()))
            else:
                losses[name] = loss_val
            total = total + w * loss_val

        losses["total"] = total
        return losses
