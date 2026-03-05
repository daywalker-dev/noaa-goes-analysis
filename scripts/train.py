#!/usr/bin/env python
"""
train.py — Main entry point for the GOES L2 forecasting pipeline.
=================================================================

Usage:
    python -m scripts.train --config config/default.yaml
    python -m scripts.train --config config/default.yaml --stage 2
    python -m scripts.train --config config/default.yaml --dry-run

Stages:
    1  Train spatial CNN encoders (land, sea, cloud)
    2  Train temporal probabilistic model (encoders frozen)
    3  Train reverse generator / conditional decoder
    4  Train fusion model

Pass ``--stage N`` to start from stage N (assumes prior stages are
checkpointed). Omit ``--stage`` to run all four stages sequentially.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── project imports (FIXED: match actual module locations) ───────────────
from utils.config_loader import load_config, validate_config, save_config
from utils.reproducibility import set_global_seed

from models.spatial_encoder import SpatialCNNEncoder, DomainEncoderEnsemble
from models.temporal_bayesian import VariationalTransformer
from models.reverse_generator import ConditionalUNet
from models.fusion import FusionTransformer

from training.stage_runners import (
    EncoderStageRunner,
    TemporalStageRunner,
    GeneratorStageRunner,
    FusionStageRunner,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ======================================================================
# Device helper
# ======================================================================

def get_device(device_str: str = "cuda") -> str:
    """Resolve device string, falling back to CPU if CUDA unavailable."""
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        return "cpu"
    return device_str


# ======================================================================
# Domain config helpers
# ======================================================================

def _get_domain_channels(cfg) -> dict[str, int]:
    """Extract {domain: n_channels} from product config."""
    domain_ch: dict[str, int] = {}
    for product in cfg.data.products:
        d = product["domain"]
        if d == "meteo":
            continue
        domain_ch[d] = domain_ch.get(d, 0) + len(product["channels"])
    return domain_ch


def _get_domain_indices(cfg) -> dict[str, list[int]]:
    """Extract {domain: [channel_indices]} from product config."""
    indices: dict[str, list[int]] = {}
    offset = 0
    for product in cfg.data.products:
        d = product["domain"]
        n = len(product["channels"])
        indices.setdefault(d, []).extend(range(offset, offset + n))
        offset += n
    return indices


def _total_channels(cfg) -> int:
    return sum(len(p["channels"]) for p in cfg.data.products)


# ======================================================================
# Synthetic data generator (for testing / dry-run)
# ======================================================================

def _make_synthetic_data(cfg):
    """
    Generate synthetic numpy arrays that mimic processed GOES L2 fields.
    Useful for smoke-testing the full pipeline without real data.
    """
    total_ch = _total_channels(cfg)
    domain_indices = _get_domain_indices(cfg)
    T_in = cfg.data.temporal.input_steps
    T_out = cfg.data.temporal.forecast_steps
    T_total = T_in + T_out
    H = W = cfg.data.spatial.patch_size
    batch_size = cfg.data.loader.batch_size
    n_train = batch_size * 10
    n_val = batch_size * 3

    meteo_idx = domain_indices.get("meteo", [])
    meteo_dim = max(len(meteo_idx), 1)

    def _make_loader(n_samples):
        data = torch.randn(n_samples, T_total, total_ch, H, W)
        mask = torch.ones(n_samples, T_out, 1, H, W)
        meteo_in = torch.randn(n_samples, T_in, meteo_dim)
        meteo_tgt = torch.randn(n_samples, T_out, meteo_dim)

        ds = TensorDataset(data, mask, meteo_in, meteo_tgt)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return _make_loader(n_train), _make_loader(n_val)


def _synthetic_batch_to_dict(batch, cfg):
    """Convert TensorDataset batch tuple into the dict format expected by stage runners."""
    data, mask, meteo_in, meteo_tgt = batch
    T_in = cfg.data.temporal.input_steps
    return {
        "input": data[:, :T_in],
        "target": data[:, T_in:],
        "mask": mask,
        "meteo_input": meteo_in,
        "meteo_target": meteo_tgt,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="GOES L2 Forecast — Training")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--stage", type=int, default=None,
                        help="Start from stage N (1-4). Default: run all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data to smoke-test the pipeline")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────
    cfg = load_config(args.config)
    validate_config(cfg)
    set_global_seed(cfg.project.get("seed", 42))
    device = get_device(cfg.project.get("device", "cuda"))
    logger.info("Device: %s", device)

    # Save a copy of the active config
    ckpt_dir = Path(cfg.project.get("checkpoint_dir", "checkpoints"))
    save_config(cfg, ckpt_dir / "config_snapshot.yaml")

    # ── Data ───────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN — using synthetic data")
        train_dl, val_dl = _make_synthetic_data(cfg)
    else:
        from data.dataset import build_dataloader
        train_dl = build_dataloader(cfg, split="train")
        val_dl = build_dataloader(cfg, split="val")

    # ── Build models ───────────────────────────────────────────────────
    domain_ch = _get_domain_channels(cfg)
    domain_idx = _get_domain_indices(cfg)
    meteo_dim = max(len(domain_idx.get("meteo", [])), 1)
    total_ch = _total_channels(cfg)

    encoders = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)

    temporal_model = VariationalTransformer(
        latent_dim=encoders.combined_latent_dim,
        meteo_dim=meteo_dim,
        state_dim=cfg.model.temporal.state_dim,
        d_model=cfg.model.temporal.d_model,
        n_heads=cfg.model.temporal.n_heads,
        n_encoder_layers=cfg.model.temporal.n_encoder_layers,
        n_decoder_layers=cfg.model.temporal.n_decoder_layers,
        dim_feedforward=cfg.model.temporal.dim_feedforward,
        dropout=cfg.model.temporal.dropout,
        forecast_steps=cfg.data.temporal.forecast_steps,
        beta_kl=cfg.model.temporal.beta_kl,
        free_bits=cfg.model.temporal.free_bits,
    )

    generator = ConditionalUNet(
        state_dim=cfg.model.temporal.state_dim,
        out_channels=total_ch,
        noise_dim=cfg.model.generator.noise_dim,
        base_channels=cfg.model.generator.base_channels,
        channel_multipliers=list(cfg.model.generator.channel_multipliers),
        n_res_blocks=cfg.model.generator.n_res_blocks,
        use_spectral_norm=cfg.model.generator.use_spectral_norm,
        initial_spatial=tuple(cfg.model.generator.initial_spatial),
    )

    fusion_model = FusionTransformer(
        cnn_dim=encoders.combined_latent_dim,
        bayes_dim=cfg.model.temporal.state_dim * 2,
        meteo_dim=meteo_dim,
        gen_dim=total_ch,
        d_model=cfg.model.fusion.d_model,
        n_heads=cfg.model.fusion.n_heads,
        n_layers=cfg.model.fusion.n_layers,
        dim_feedforward=cfg.model.fusion.dim_feedforward,
        dropout=cfg.model.fusion.dropout,
        out_channels=total_ch,
        spatial_size=(cfg.data.spatial.patch_size, cfg.data.spatial.patch_size),
    )

    start = args.stage or 1

    # ── Stage 1 ─────────────────────────────────────────────────────
    if start <= 1:
        logger.info("═══ STAGE 1: Training Spatial Encoders ═══")
        runner = EncoderStageRunner(
            model=encoders,
            train_loader=train_dl,
            val_loader=val_dl,
            cfg=cfg,
            output_dir=ckpt_dir / "encoders",
            domain_indices=domain_idx,
        )
        runner.run()
    else:
        ckpt = ckpt_dir / "encoders" / "best.pt"
        if ckpt.exists():
            encoders.load_state_dict(torch.load(ckpt, weights_only=True))
            encoders.to(device)
            logger.info("Loaded encoder checkpoint: %s", ckpt)

    # ── Stage 2 ─────────────────────────────────────────────────────
    if start <= 2:
        logger.info("═══ STAGE 2: Training Temporal Model ═══")
        runner = TemporalStageRunner(
            model=temporal_model,
            encoders=encoders,
            train_loader=train_dl,
            val_loader=val_dl,
            cfg=cfg,
            output_dir=ckpt_dir / "temporal",
            domain_indices=domain_idx,
        )
        runner.run()
    else:
        ckpt = ckpt_dir / "temporal" / "best.pt"
        if ckpt.exists():
            temporal_model.load_state_dict(torch.load(ckpt, weights_only=True))
            temporal_model.to(device)

    # ── Stage 3 ─────────────────────────────────────────────────────
    if start <= 3:
        logger.info("═══ STAGE 3: Training Reverse Generator ═══")
        runner = GeneratorStageRunner(
            model=generator,
            encoders=encoders,
            temporal_model=temporal_model,
            train_loader=train_dl,
            val_loader=val_dl,
            cfg=cfg,
            output_dir=ckpt_dir / "generator",
            domain_indices=domain_idx,
        )
        runner.run()
    else:
        ckpt = ckpt_dir / "generator" / "best.pt"
        if ckpt.exists():
            generator.load_state_dict(torch.load(ckpt, weights_only=True))
            generator.to(device)

    # ── Stage 4 ─────────────────────────────────────────────────────
    if start <= 4:
        logger.info("═══ STAGE 4: Training Fusion Model ═══")
        runner = FusionStageRunner(
            model=fusion_model,
            encoders=encoders,
            temporal_model=temporal_model,
            generator=generator,
            train_loader=train_dl,
            val_loader=val_dl,
            cfg=cfg,
            output_dir=ckpt_dir / "fusion",
            domain_indices=domain_idx,
        )
        runner.run()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()