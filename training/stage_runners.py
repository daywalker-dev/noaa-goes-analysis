"""Stage-specific training runners for the 4-stage pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from goes_forecast.training.trainer import BaseTrainer
from goes_forecast.training.losses import (
    MaskedMSE, SSIMLoss, CRPSLoss, SpectralLoss,
    PhysicsConstraintLoss, ELBOLoss,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _split_domains(
    batch: dict, domain_indices: dict[str, list[int]]
) -> dict[str, torch.Tensor]:
    """Split input tensor by domain channel indices."""
    x = batch["input"]  # (B, T, C, H, W)
    domains = {}
    for domain, idx in domain_indices.items():
        if idx and domain != "meteo":
            domains[domain] = x[:, :, idx]
    return domains


# ---------------------------------------------------------------------------
# Stage 1: Encoder Stage Runner
# ---------------------------------------------------------------------------
class EncoderStageRunner(BaseTrainer):
    """Trains the DomainEncoderEnsemble on next-step reconstruction.

    Loss = MSE + SSIM + Physics (weighted from config).
    """

    def __init__(
        self,
        model,  # DomainEncoderEnsemble
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        output_dir: str | Path,
        domain_indices: dict[str, list[int]],
        **kwargs,
    ):
        stage_cfg = cfg.training.stages.encoders
        super().__init__(model, train_loader, val_loader, cfg, stage_cfg, output_dir, **kwargs)
        self.domain_indices = domain_indices
        weights = dict(stage_cfg.loss_weights)

        self.mse_loss = MaskedMSE()
        self.ssim_loss = SSIMLoss()
        self.physics_loss = PhysicsConstraintLoss()
        self.weights = weights

    def _compute_loss(
        self, recon: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        mse = self.mse_loss(recon, target, mask)
        ssim = self.ssim_loss(recon, target) if recon.shape[-1] >= 7 else torch.tensor(0.0, device=recon.device)
        physics = self.physics_loss(recon, target)

        total = (
            self.weights.get("mse", 1.0) * mse
            + self.weights.get("ssim", 0.5) * ssim
            + self.weights.get("physics", 0.1) * physics
        )
        return {"total": total, "mse": mse, "ssim": ssim, "physics": physics}

    def _step(self, batch: dict) -> dict[str, torch.Tensor]:
        B, T, C, H, W = batch["input"].shape

        # Use first T-1 steps as input, T as target (next-step prediction)
        input_data = batch["input"][:, :-1]  # (B, T-1, C, H, W)
        target_data = batch["input"][:, 1:]  # (B, T-1, C, H, W) — shifted by 1

        all_losses = {"total": torch.tensor(0.0, device=self.device)}

        for domain, idx in self.domain_indices.items():
            if not idx or domain == "meteo":
                continue

            # Flatten time into batch for encoder: (B*(T-1), C_domain, H, W)
            domain_in = input_data[:, :, idx].reshape(-1, len(idx), H, W)
            domain_tgt = target_data[:, :, idx].reshape(-1, len(idx), H, W)

            results = self.model.encoders[domain](domain_in)
            losses = self._compute_loss(results["reconstruction"], domain_tgt, None)

            for k, v in losses.items():
                all_losses[k] = all_losses.get(k, torch.tensor(0.0, device=self.device)) + v

        return all_losses

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def _val_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)


# ---------------------------------------------------------------------------
# Stage 2: Temporal Stage Runner
# ---------------------------------------------------------------------------
class TemporalStageRunner(BaseTrainer):
    """Trains the VariationalTransformer with frozen encoders.

    Loss = ELBO (recon + β·KL) + CRPS.
    """

    def __init__(
        self,
        model,  # VariationalTransformer
        encoders,  # DomainEncoderEnsemble (frozen)
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        output_dir: str | Path,
        domain_indices: dict[str, list[int]],
        **kwargs,
    ):
        stage_cfg = cfg.training.stages.temporal
        super().__init__(model, train_loader, val_loader, cfg, stage_cfg, output_dir, **kwargs)
        self.encoders = encoders.to(self.device)
        self.encoders.freeze()
        self.domain_indices = domain_indices
        weights = dict(stage_cfg.loss_weights)

        self.elbo_loss = ELBOLoss(
            beta=cfg.model.temporal.beta_kl,
            free_bits=cfg.model.temporal.free_bits,
        )
        self.crps_loss = CRPSLoss()
        self.weights = weights
        self.kl_anneal_epochs = cfg.model.temporal.kl_anneal_epochs

    @torch.no_grad()
    def _extract_latents(self, batch: dict) -> torch.Tensor:
        """Run frozen encoders on input to get latent time series."""
        x = batch["input"]  # (B, T, C, H, W)
        B, T, C, H, W = x.shape
        all_latents = []

        for t in range(T):
            frame = x[:, t]  # (B, C, H, W)
            domain_inputs = {}
            for domain, idx in self.domain_indices.items():
                if idx and domain != "meteo":
                    domain_inputs[domain] = frame[:, idx]
            latent = self.encoders.encode_only(domain_inputs)
            all_latents.append(latent)

        return torch.stack(all_latents, dim=1)  # (B, T, D)

    def _build_state_target(self, batch: dict) -> torch.Tensor:
        """Build target state vector from spatially-averaged L2 channels."""
        target = batch["target"]  # (B, T_out, C, H, W)
        return target.mean(dim=(-2, -1))  # (B, T_out, C)

    def _step(self, batch: dict, epoch: int = 0) -> dict[str, torch.Tensor]:
        latents = self._extract_latents(batch)
        meteo = batch["meteo_input"]
        state_target = self._build_state_target(batch)

        out = self.model(latents, meteo)
        mu, logvar = out["mu"], out["logvar"]

        # Align dimensions: state_target may differ from state_dim
        if mu.shape[-1] != state_target.shape[-1]:
            state_target = state_target[..., :mu.shape[-1]]

        # KL annealing
        beta = min(1.0, epoch / max(self.kl_anneal_epochs, 1)) * self.elbo_loss.beta
        self.elbo_loss.set_beta(beta)

        elbo = self.elbo_loss(mu, logvar, state_target)
        crps = self.crps_loss(mu, logvar, state_target)

        total = (
            self.weights.get("elbo", 1.0) * elbo["total"]
            + self.weights.get("crps", 0.5) * crps
        )
        return {"total": total, "recon": elbo["recon"], "kl": elbo["kl"], "crps": crps}

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def _val_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)


# ---------------------------------------------------------------------------
# Stage 3: Generator Stage Runner
# ---------------------------------------------------------------------------
class GeneratorStageRunner(BaseTrainer):
    """Trains the ConditionalUNet with frozen encoder + temporal model.

    Loss = MSE + SSIM + Spectral.
    """

    def __init__(
        self,
        model,  # ConditionalUNet
        encoders,  # frozen
        temporal_model,  # frozen VariationalTransformer
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        output_dir: str | Path,
        domain_indices: dict[str, list[int]],
        **kwargs,
    ):
        stage_cfg = cfg.training.stages.generator
        super().__init__(model, train_loader, val_loader, cfg, stage_cfg, output_dir, **kwargs)
        self.encoders = encoders.to(self.device)
        self.temporal_model = temporal_model.to(self.device)
        self.encoders.freeze()
        for p in self.temporal_model.parameters():
            p.requires_grad = False
        self.domain_indices = domain_indices
        weights = dict(stage_cfg.loss_weights)

        self.mse_loss = MaskedMSE()
        self.ssim_loss = SSIMLoss()
        self.spectral_loss = SpectralLoss()
        self.weights = weights

    @torch.no_grad()
    def _get_predicted_states(self, batch: dict) -> torch.Tensor:
        """Run frozen encoder + temporal to get predicted states."""
        x = batch["input"]
        B, T, C, H, W = x.shape
        latents = []
        for t in range(T):
            frame = x[:, t]
            domain_inputs = {
                d: frame[:, idx]
                for d, idx in self.domain_indices.items()
                if idx and d != "meteo"
            }
            latents.append(self.encoders.encode_only(domain_inputs))
        latents = torch.stack(latents, dim=1)
        meteo = batch["meteo_input"]

        out = self.temporal_model(latents, meteo)
        return out["mu"]  # (B, T_out, state_dim)

    def _step(self, batch: dict) -> dict[str, torch.Tensor]:
        states = self._get_predicted_states(batch)
        target = batch["target"]  # (B, T_out, C, H, W)
        B, T, C, H, W = target.shape

        # Decode all timesteps
        generated = self.model.decode_sequence(states, target_size=(H, W))
        gen_flat = generated.reshape(B * T, C, H, W)
        tgt_flat = target.reshape(B * T, C, H, W)

        mse = self.mse_loss(gen_flat, tgt_flat)
        ssim = self.ssim_loss(gen_flat, tgt_flat) if H >= 7 else torch.tensor(0.0, device=self.device)
        spectral = self.spectral_loss(gen_flat, tgt_flat)

        total = (
            self.weights.get("mse", 1.0) * mse
            + self.weights.get("ssim", 0.5) * ssim
            + self.weights.get("spectral", 0.2) * spectral
        )
        return {"total": total, "mse": mse, "ssim": ssim, "spectral": spectral}

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def _val_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)


# ---------------------------------------------------------------------------
# Stage 4: Fusion Stage Runner
# ---------------------------------------------------------------------------
class FusionStageRunner(BaseTrainer):
    """Trains the FusionTransformer, optionally unfreezing all sub-models."""

    def __init__(
        self,
        model,  # FusionTransformer
        encoders,
        temporal_model,
        generator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        output_dir: str | Path,
        domain_indices: dict[str, list[int]],
        **kwargs,
    ):
        stage_cfg = cfg.training.stages.fusion

        # Optionally unfreeze all
        if stage_cfg.get("unfreeze_all", False):
            encoders.unfreeze()
            for p in temporal_model.parameters():
                p.requires_grad = True
            for p in generator.parameters():
                p.requires_grad = True

        # Wrap all models in a combined module for optimizer
        self.all_models = nn.ModuleDict({
            "fusion": model,
            "encoders": encoders,
            "temporal": temporal_model,
            "generator": generator,
        })
        super().__init__(
            self.all_models, train_loader, val_loader, cfg, stage_cfg, output_dir, **kwargs
        )
        self.encoders = encoders
        self.temporal_model = temporal_model
        self.generator = generator
        self.fusion = model
        self.domain_indices = domain_indices
        weights = dict(stage_cfg.loss_weights)

        self.mse_loss = MaskedMSE()
        self.ssim_loss = SSIMLoss()
        self.crps_loss = CRPSLoss()
        self.physics_loss = PhysicsConstraintLoss()
        self.weights = weights

    def _forward_all(self, batch: dict) -> dict[str, torch.Tensor]:
        """Run full pipeline: encoder → temporal → generator → fusion."""
        x = batch["input"]
        B, T_in, C, H, W = x.shape
        T_out = batch["target"].shape[1]

        # Extract latents per timestep
        latents = []
        for t in range(T_in):
            frame = x[:, t]
            domain_inputs = {
                d: frame[:, idx]
                for d, idx in self.domain_indices.items()
                if idx and d != "meteo"
            }
            latents.append(self.encoders.encode_only(domain_inputs))
        latents_stack = torch.stack(latents, dim=1)  # (B, T_in, D_enc)

        # Temporal model
        meteo_in = batch["meteo_input"]
        temporal_out = self.temporal_model(latents_stack, meteo_in)
        mu, logvar = temporal_out["mu"], temporal_out["logvar"]

        # Generator
        gen_spatial = self.generator.decode_sequence(mu, target_size=(H, W))

        # Repeat last encoder latent for forecast steps
        last_latent = latents_stack[:, -1:].expand(-1, T_out, -1)

        # Bayesian features
        bayes_feat = torch.cat([mu, logvar], dim=-1)

        # Generator summary features (spatial avg)
        gen_feat = gen_spatial.mean(dim=(-2, -1))  # (B, T, C)

        # Meteo target
        meteo_tgt = batch["meteo_target"]

        # Fusion
        fusion_out = self.fusion(
            cnn_latents=last_latent,
            bayes_output=bayes_feat,
            meteo_fields=meteo_tgt,
            gen_features=gen_feat,
            gen_spatial=gen_spatial,
        )
        return {**fusion_out, "temporal_mu": mu, "temporal_logvar": logvar, "gen_spatial": gen_spatial}

    def _step(self, batch: dict) -> dict[str, torch.Tensor]:
        out = self._forward_all(batch)
        target = batch["target"]
        B, T, C, H, W = target.shape

        if "forecast" in out:
            pred = out["forecast"]
            pred_flat = pred.reshape(B * T, C, H, W)
            tgt_flat = target.reshape(B * T, C, H, W)
            mse = self.mse_loss(pred_flat, tgt_flat)
            ssim = self.ssim_loss(pred_flat, tgt_flat) if H >= 7 else torch.tensor(0.0, device=self.device)
        else:
            mse = torch.tensor(0.0, device=self.device)
            ssim = torch.tensor(0.0, device=self.device)

        # CRPS on temporal output
        state_target = target.mean(dim=(-2, -1))
        mu, lv = out["temporal_mu"], out["temporal_logvar"]
        if mu.shape[-1] != state_target.shape[-1]:
            state_target = state_target[..., :mu.shape[-1]]
        crps = self.crps_loss(mu, lv, state_target)

        physics = self.physics_loss(out.get("forecast", out["gen_spatial"]), target)

        total = (
            self.weights.get("mse", 1.0) * mse
            + self.weights.get("ssim", 0.5) * ssim
            + self.weights.get("crps", 0.5) * crps
            + self.weights.get("physics", 0.1) * physics
        )
        return {"total": total, "mse": mse, "ssim": ssim, "crps": crps, "physics": physics}

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)

    def _val_step(self, batch: dict) -> dict[str, torch.Tensor]:
        return self._step(batch)
