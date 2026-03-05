"""
Multi-Stage Training Engine
============================
Implements the four-stage training pipeline:

    Stage 1 — Spatial CNN encoders (land, sea, cloud)
    Stage 2 — Temporal probabilistic model (encoders frozen)
    Stage 3 — Reverse generator / conditional decoder
    Stage 4 — Full fusion model

Each stage handles its own optimizer, scheduler, loss, checkpointing,
and mixed-precision context.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from models.blocks import CombinedLoss
from utils.reproducibility import amp_context, EarlyStopping

logger = logging.getLogger(__name__)


# ===========================================================================
# Scheduler factory
# ===========================================================================

def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    total_epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    name = cfg.get("scheduler", "cosine")
    if name == "cosine":
        warmup = cfg.get("warmup_epochs", 0)
        return CosineAnnealingLR(optimizer, T_max=total_epochs - warmup)
    elif name == "step":
        return StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])
    return None


# ===========================================================================
# Generic epoch runner
# ===========================================================================

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    mixed_precision: bool,
    scaler: Optional[torch.amp.GradScaler],
    prepare_batch_fn: Any,
    is_train: bool = True,
) -> float:
    """Run one epoch of training or validation.

    *prepare_batch_fn* is a callable that receives a batch dict and device
    and returns ``(model_input, target, mask, prev)`` tuples.
    """
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            inputs, target, mask, prev = prepare_batch_fn(batch, device)
            with torch.amp.autocast("cuda", enabled=mixed_precision and device.type == "cuda"):
                pred = model(inputs)
                loss = criterion(pred, target, mask, prev)

            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * target.shape[0]
            n += target.shape[0]

    return total_loss / max(n, 1)


# ===========================================================================
# Stage 1 — Encoder Training
# ===========================================================================

def train_encoder(
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
    tag: str = "encoder",
) -> nn.Module:
    """Train a single spatial CNN encoder."""
    stage_cfg = cfg["training"]["stage_1_encoder"]
    loss_cfg = stage_cfg.get("loss", {})

    criterion = CombinedLoss(
        mse_weight=loss_cfg.get("mse_weight", 1.0),
        ssim_weight=loss_cfg.get("ssim_weight", 0.3),
        physics_weight=loss_cfg.get("physics_weight", 0.0),
    ).to(device)

    encoder = encoder.to(device)
    optimizer = AdamW(encoder.parameters(), lr=stage_cfg["lr"],
                      weight_decay=stage_cfg.get("weight_decay", 1e-5))
    scheduler = _build_scheduler(optimizer, stage_cfg, stage_cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda") if cfg["project"].get("mixed_precision") else None
    stopper = EarlyStopping(patience=12)

    ckpt_dir = Path(cfg["project"]["checkpoint_dir"]) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _prep(batch: Dict, dev: torch.device) -> Tuple:
        # Use land/sea/cloud inputs depending on tag
        key_in = f"{tag.split('_')[0]}_in"
        key_tgt = f"{tag.split('_')[0]}_target"
        x = batch.get(key_in, batch.get("land_in"))
        # Collapse time into batch: (B, T, C, H, W) → (B*T, C, H, W)
        B, T = x.shape[:2]
        x = x.view(B * T, *x.shape[2:]).to(dev)
        tgt = batch.get(key_tgt, batch.get("land_target"))
        tgt = tgt[:, 0].to(dev)  # first future step
        # Expand target to match encoder output channels
        tgt = tgt[:, :x.shape[1]]  # trim channels if needed
        mask = batch["mask_target"][:, 0:1].to(dev) if "mask_target" in batch else None
        return x[:B], tgt, mask, None  # use last input frame

    best_val = float("inf")
    for epoch in range(stage_cfg["epochs"]):
        t0 = time.time()
        train_loss = _run_epoch(
            _EncoderWrapper(encoder), train_loader, criterion,
            optimizer, device, cfg["project"].get("mixed_precision", False),
            scaler, _prep, is_train=True,
        )
        val_loss = _run_epoch(
            _EncoderWrapper(encoder), val_loader, criterion,
            None, device, cfg["project"].get("mixed_precision", False),
            None, _prep, is_train=False,
        )
        if scheduler:
            scheduler.step()

        dt = time.time() - t0
        logger.info("[%s] Epoch %03d  train=%.5f  val=%.5f  (%.1fs)",
                     tag, epoch + 1, train_loss, val_loss, dt)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(encoder.state_dict(), ckpt_dir / "best.pt")

        if stopper.step(val_loss):
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Reload best weights
    encoder.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    return encoder


class _EncoderWrapper(nn.Module):
    """Thin wrapper so the generic epoch runner gets spatial prediction output."""
    def __init__(self, enc: nn.Module) -> None:
        super().__init__()
        self.enc = enc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, pred, _ = self.enc(x)
        return pred


# ===========================================================================
# Stage 2 — Temporal Model Training
# ===========================================================================

def train_temporal(
    temporal_model: nn.Module,
    encoders: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Train the probabilistic temporal model with frozen encoders."""
    stage_cfg = cfg["training"]["stage_2_temporal"]

    # Freeze encoders
    if stage_cfg.get("freeze_encoders", True):
        for enc in encoders.values():
            enc.eval()
            for p in enc.parameters():
                p.requires_grad_(False)

    temporal_model = temporal_model.to(device)
    optimizer = AdamW(temporal_model.parameters(), lr=stage_cfg["lr"],
                      weight_decay=stage_cfg.get("weight_decay", 1e-5))
    scheduler = _build_scheduler(optimizer, stage_cfg, stage_cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda") if cfg["project"].get("mixed_precision") else None
    stopper = EarlyStopping(patience=15)

    kl_weight = cfg["temporal"].get("kl_weight", 0.001)

    ckpt_dir = Path(cfg["project"]["checkpoint_dir"]) / "temporal"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(stage_cfg["epochs"]):
        temporal_model.train()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            # Encode each group
            latents = []
            for key in ["land", "sea", "cloud"]:
                enc = encoders.get(key)
                if enc is None:
                    continue
                x = batch[f"{key}_in"].to(device)
                B, T = x.shape[:2]
                x_flat = x.view(B * T, *x.shape[2:])
                with torch.no_grad():
                    z, _, _ = enc(x_flat)
                z = z.view(B, T, -1)
                latents.append(z)

            # Append auxiliary features (flattened spatially)
            for aux_key in ["wind_in", "thermo_in"]:
                a = batch[aux_key].to(device)
                B, T = a.shape[:2]
                a_flat = a.view(B, T, -1)[:, :, :32]  # truncate to manageable size
                latents.append(a_flat)

            x_seq = torch.cat(latents, dim=-1)  # (B, T, D)

            # Pad/truncate to expected input_dim
            D = cfg["temporal"]["input_dim"]
            if x_seq.shape[-1] < D:
                pad = torch.zeros(*x_seq.shape[:2], D - x_seq.shape[-1], device=device)
                x_seq = torch.cat([x_seq, pad], dim=-1)
            else:
                x_seq = x_seq[:, :, :D]

            with torch.amp.autocast("cuda", enabled=cfg["project"].get("mixed_precision", False)):
                out = temporal_model(x_seq)
                recon_loss = nn.functional.mse_loss(out["pred_mu"], x_seq)
                kl = out["kl_loss"]
                loss = recon_loss + kl_weight * kl

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * B
            n += B

        if scheduler:
            scheduler.step()

        epoch_loss = total_loss / max(n, 1)
        logger.info("[temporal] Epoch %03d  loss=%.5f", epoch + 1, epoch_loss)

        if epoch_loss < best_val:
            best_val = epoch_loss
            torch.save(temporal_model.state_dict(), ckpt_dir / "best.pt")

        if stopper.step(epoch_loss):
            logger.info("Early stopping temporal at epoch %d", epoch + 1)
            break

    temporal_model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    return temporal_model


# ===========================================================================
# Stage 3 — Decoder Training
# ===========================================================================

def train_decoder(
    decoder: nn.Module,
    temporal_model: nn.Module,
    encoders: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Train the reverse generator conditioned on temporal model outputs."""
    stage_cfg = cfg["training"]["stage_3_decoder"]

    decoder = decoder.to(device)
    temporal_model.eval()
    for p in temporal_model.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(decoder.parameters(), lr=stage_cfg["lr"],
                      weight_decay=stage_cfg.get("weight_decay", 1e-5))
    scheduler = _build_scheduler(optimizer, stage_cfg, stage_cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda") if cfg["project"].get("mixed_precision") else None
    stopper = EarlyStopping(patience=10)

    criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.3).to(device)

    ckpt_dir = Path(cfg["project"]["checkpoint_dir"]) / "decoder"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(stage_cfg["epochs"]):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in train_loader:
            # Get temporal latent
            latents = []
            for key in ["land", "sea", "cloud"]:
                enc = encoders.get(key)
                if enc is None:
                    continue
                x = batch[f"{key}_in"].to(device)
                B, T = x.shape[:2]
                x_flat = x.view(B * T, *x.shape[2:])
                with torch.no_grad():
                    z, _, _ = enc(x_flat)
                z = z.view(B, T, -1)
                latents.append(z)

            x_seq = torch.cat(latents, dim=-1)
            D = cfg["temporal"]["input_dim"]
            if x_seq.shape[-1] < D:
                pad = torch.zeros(*x_seq.shape[:2], D - x_seq.shape[-1], device=device)
                x_seq = torch.cat([x_seq, pad], dim=-1)
            else:
                x_seq = x_seq[:, :, :D]

            with torch.no_grad():
                temp_out = temporal_model(x_seq)
            z_pred = temp_out["pred_mu"][:, -1]  # last-step prediction

            # Truncate to decoder input dim
            dec_dim = cfg["decoder"]["in_channels"]
            z_dec = z_pred[:, :dec_dim]

            # Target: all available L2 channels stacked
            targets = []
            for key in ["land_target", "sea_target", "cloud_target"]:
                t = batch[key][:, 0].to(device)  # first future step
                targets.append(t)
            target = torch.cat(targets, dim=1)

            # Trim to decoder output channels
            out_ch = cfg["decoder"]["out_channels"]
            target = target[:, :out_ch]

            with torch.amp.autocast("cuda", enabled=cfg["project"].get("mixed_precision", False)):
                recon = decoder(z_dec)
                loss = criterion(recon, target)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * z_dec.shape[0]
            n += z_dec.shape[0]

        if scheduler:
            scheduler.step()

        epoch_loss = total_loss / max(n, 1)
        logger.info("[decoder] Epoch %03d  loss=%.5f", epoch + 1, epoch_loss)

        if epoch_loss < best_val:
            best_val = epoch_loss
            torch.save(decoder.state_dict(), ckpt_dir / "best.pt")

        if stopper.step(epoch_loss):
            break

    decoder.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    return decoder


# ===========================================================================
# Stage 4 — Fusion Training
# ===========================================================================

def train_fusion(
    fusion_model: nn.Module,
    decoder: nn.Module,
    temporal_model: nn.Module,
    encoders: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict[str, Any],
    device: torch.device,
) -> nn.Module:
    """Train the full fusion model combining all streams."""
    stage_cfg = cfg["training"]["stage_4_fusion"]

    fusion_model = fusion_model.to(device)
    # Freeze upstream
    for m in [temporal_model, decoder, *encoders.values()]:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    optimizer = AdamW(fusion_model.parameters(), lr=stage_cfg["lr"],
                      weight_decay=stage_cfg.get("weight_decay", 1e-5))
    scheduler = _build_scheduler(optimizer, stage_cfg, stage_cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda") if cfg["project"].get("mixed_precision") else None
    stopper = EarlyStopping(patience=10)
    criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.2).to(device)

    ckpt_dir = Path(cfg["project"]["checkpoint_dir"]) / "fusion"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    hidden_dim = cfg["fusion"]["hidden_dim"]

    for epoch in range(stage_cfg["epochs"]):
        fusion_model.train()
        total_loss, n = 0.0, 0

        for batch in train_loader:
            sources = []

            # Source 1: original L2 (global avg pool → vector)
            orig_fields = []
            for key in ["land_in", "sea_in", "cloud_in"]:
                f = batch[key][:, -1].to(device)  # last input frame
                orig_fields.append(f)
            orig = torch.cat(orig_fields, dim=1)
            orig_vec = orig.mean(dim=(-2, -1))  # (B, C_total)
            # Project to hidden_dim
            if not hasattr(train_fusion, "_orig_proj"):
                train_fusion._orig_proj = nn.Linear(orig_vec.shape[-1], hidden_dim).to(device)
            sources.append(train_fusion._orig_proj(orig_vec))

            # Source 2: CNN encoder predictions
            enc_preds = []
            for key in ["land", "sea", "cloud"]:
                enc = encoders.get(key)
                if enc is None:
                    continue
                x = batch[f"{key}_in"][:, -1].to(device)
                with torch.no_grad():
                    z, pred, _ = enc(x)
                enc_preds.append(z)
            enc_cat = torch.cat(enc_preds, dim=-1) if enc_preds else torch.zeros(orig.shape[0], hidden_dim, device=device)
            if not hasattr(train_fusion, "_enc_proj"):
                train_fusion._enc_proj = nn.Linear(enc_cat.shape[-1], hidden_dim).to(device)
            sources.append(train_fusion._enc_proj(enc_cat))

            # Source 3: temporal model output
            latents = []
            for key in ["land", "sea", "cloud"]:
                enc = encoders.get(key)
                if enc is None:
                    continue
                x = batch[f"{key}_in"].to(device)
                B, T = x.shape[:2]
                x_flat = x.view(B * T, *x.shape[2:])
                with torch.no_grad():
                    z, _, _ = enc(x_flat)
                z = z.view(B, T, -1)
                latents.append(z)
            x_seq = torch.cat(latents, dim=-1)
            D = cfg["temporal"]["input_dim"]
            if x_seq.shape[-1] < D:
                pad = torch.zeros(*x_seq.shape[:2], D - x_seq.shape[-1], device=device)
                x_seq = torch.cat([x_seq, pad], dim=-1)
            else:
                x_seq = x_seq[:, :, :D]
            with torch.no_grad():
                temp_out = temporal_model(x_seq)
            temp_vec = temp_out["pred_mu"][:, -1, :hidden_dim]
            if temp_vec.shape[-1] < hidden_dim:
                temp_vec = F.pad(temp_vec, (0, hidden_dim - temp_vec.shape[-1]))
            sources.append(temp_vec)

            # Source 4: wind + humidity aux
            wind = batch["wind_in"][:, -1].to(device).mean(dim=(-2, -1))
            thermo = batch["thermo_in"][:, -1].to(device).mean(dim=(-2, -1))
            aux = torch.cat([wind, thermo], dim=-1)
            if not hasattr(train_fusion, "_aux_proj"):
                train_fusion._aux_proj = nn.Linear(aux.shape[-1], hidden_dim).to(device)
            sources.append(train_fusion._aux_proj(aux))

            # Target
            targets = []
            for key in ["land_target", "sea_target", "cloud_target"]:
                targets.append(batch[key][:, 0].to(device))
            target = torch.cat(targets, dim=1)[:, :cfg["fusion"]["output_dim"]]

            with torch.amp.autocast("cuda", enabled=cfg["project"].get("mixed_precision", False)):
                out = fusion_model(sources)
                loss = criterion(out, target)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * target.shape[0]
            n += target.shape[0]

        if scheduler:
            scheduler.step()

        epoch_loss = total_loss / max(n, 1)
        logger.info("[fusion] Epoch %03d  loss=%.5f", epoch + 1, epoch_loss)

        if epoch_loss < best_val:
            best_val = epoch_loss
            torch.save(fusion_model.state_dict(), ckpt_dir / "best.pt")

        if stopper.step(epoch_loss):
            break

    fusion_model.load_state_dict(torch.load(ckpt_dir / "best.pt", weights_only=True))
    return fusion_model