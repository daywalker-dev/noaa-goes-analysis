#!/usr/bin/env python
"""
train.py — Main entry point for the GOES L2 forecasting pipeline.
=================================================================

Usage:
    python -m scripts.train --config configs/default.yaml
    python -m scripts.train --config configs/default.yaml --stage 2
    python -m scripts.train --config configs/default.yaml --dry-run

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

import torch

# ── project imports ──────────────────────────────────────────────────────
from utils.config import load_config, validate_config, save_config
from utils.reproducibility import seed_everything, get_device

from models.encoders.spatial_cnn import build_encoder
from models.temporal.probabilistic import build_temporal_model
from models.decoder.reverse_generator import build_decoder
from models.fusion.fusion_model import build_fusion_model

from training.trainer import (
    train_encoder,
    train_temporal,
    train_decoder,
    train_fusion,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ======================================================================
# Synthetic data generator (for testing / dry-run)
# ======================================================================

def _make_synthetic_data(cfg: Dict[str, Any]):
    """
    Generate synthetic numpy arrays that mimic processed GOES L2 fields.
    Useful for smoke-testing the full pipeline without real data.
    """
    import numpy as np
    from data.dataset import build_dataloaders

    grid = cfg["data"]["spatial"]["grid_size"]
    H, W = grid
    T = 200  # synthetic time-steps

    data_arrays: Dict[str, np.ndarray] = {}
    masks: Dict[str, np.ndarray] = {}

    # Land surface temperature
    data_arrays["LST"] = np.random.randn(T, H, W).astype(np.float32) * 5 + 300
    masks["LST"] = np.ones((T, H, W), dtype=bool)

    # Sea surface temperature
    data_arrays["SST"] = np.random.randn(T, H, W).astype(np.float32) * 2 + 290
    masks["SST"] = np.ones((T, H, W), dtype=bool)

    # Cloud / moisture
    data_arrays["CMI"] = np.random.rand(T, H, W).astype(np.float32)
    masks["CMI"] = np.ones((T, H, W), dtype=bool)

    # Total precipitable water
    data_arrays["TPW"] = np.random.rand(T, H, W).astype(np.float32) * 60
    masks["TPW"] = np.ones((T, H, W), dtype=bool)

    # Derived motion winds
    data_arrays["wind_speed"] = np.random.rand(T, H, W).astype(np.float32) * 30
    data_arrays["wind_direction"] = np.random.rand(T, H, W).astype(np.float32) * 360
    masks["wind_speed"] = np.ones((T, H, W), dtype=bool)
    masks["wind_direction"] = np.ones((T, H, W), dtype=bool)

    # Inject some missing data
    for key in data_arrays:
        miss = np.random.rand(T, H, W) < 0.02
        data_arrays[key][miss] = np.nan
        masks[key][miss] = False

    # Fill missing
    for key in data_arrays:
        arr = data_arrays[key]
        arr[~np.isfinite(arr)] = 0.0

    train_dl, val_dl = build_dataloaders(data_arrays, masks, cfg)
    return train_dl, val_dl


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="GOES L2 Forecast — Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--stage", type=int, default=None,
                        help="Start from this stage (1–4). Default: run all.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use synthetic data to smoke-test the pipeline")
    args = parser.parse_args()

    # ── Load config ────────────────────────────────────────────────────
    cfg = load_config(args.config)
    validate_config(cfg)
    seed_everything(cfg["project"].get("seed", 42))
    device = get_device(cfg["project"].get("device", "cuda"))
    logger.info("Device: %s", device)

    # Save a copy of the active config
    save_config(cfg, Path(cfg["project"]["checkpoint_dir"]) / "config_snapshot.yaml")

    # ── Data ───────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY RUN — using synthetic data")
        train_dl, val_dl = _make_synthetic_data(cfg)
    else:
        # Real data pipeline
        from data.ingest import (
            download_all_products,
            reproject_to_common_grid,
            align_temporal,
            FieldNormalizer,
            fill_missing,
        )
        from data.dataset import build_dataloaders
        import numpy as np

        logger.info("Downloading GOES L2 products...")
        raw = download_all_products(cfg)

        logger.info("Reprojecting to common grid...")
        grid = tuple(cfg["data"]["spatial"]["grid_size"])
        for prod, ds_list in raw.items():
            raw[prod] = [reproject_to_common_grid(ds, grid) for ds in ds_list]

        logger.info("Temporal alignment...")
        aligned = align_temporal(raw, cfg["data"]["temporal"]["interval_hours"])

        # Convert to numpy arrays  (T, H, W) per variable
        data_arrays = {}
        masks = {}
        for prod, ds in aligned.items():
            from data.ingest import PRODUCT_VARIABLE_MAP
            for var in PRODUCT_VARIABLE_MAP.get(prod, []):
                if var in ds:
                    arr = ds[var].values.astype(np.float32)
                    filled, msk = fill_missing(
                        arr[:, np.newaxis],
                        cfg["data"]["missing_data"]["fill_strategy"],
                    )
                    data_arrays[var] = filled[:, 0]
                    masks[var] = msk[:, 0]

        # Normalize
        normalizer = FieldNormalizer(cfg["data"]["normalization"]["method"])
        all_data = np.stack(list(data_arrays.values()), axis=1)
        normalizer.fit(all_data, list(data_arrays.keys()))
        normed = normalizer.transform(all_data, list(data_arrays.keys()))
        for i, key in enumerate(data_arrays):
            data_arrays[key] = normed[:, i]

        train_dl, val_dl = build_dataloaders(data_arrays, masks, cfg)

    # ── Build models ───────────────────────────────────────────────────
    grid_size = tuple(cfg["data"]["spatial"]["grid_size"])
    encoders = {
        "land": build_encoder(cfg["encoder"]["land"]),
        "sea": build_encoder(cfg["encoder"]["sea"]),
        "cloud": build_encoder(cfg["encoder"]["cloud"]),
    }
    temporal_model = build_temporal_model(cfg["temporal"])
    decoder = build_decoder(cfg["decoder"], grid_size)
    fusion_model = build_fusion_model(cfg["fusion"], grid_size)

    start = args.stage or 1

    # ── Stage 1 ─────────────────────────────────────────────────────
    if start <= 1:
        logger.info("═══ STAGE 1: Training Spatial Encoders ═══")
        for name, enc in encoders.items():
            logger.info("Training encoder: %s", name)
            encoders[name] = train_encoder(
                enc, train_dl, val_dl, cfg, device, tag=f"{name}_encoder",
            )
    else:
        # Load checkpointed encoders
        for name, enc in encoders.items():
            ckpt = Path(cfg["project"]["checkpoint_dir"]) / f"{name}_encoder" / "best.pt"
            if ckpt.exists():
                enc.load_state_dict(torch.load(ckpt, weights_only=True))
                enc.to(device)
                logger.info("Loaded encoder checkpoint: %s", ckpt)

    # ── Stage 2 ─────────────────────────────────────────────────────
    if start <= 2:
        logger.info("═══ STAGE 2: Training Temporal Model ═══")
        temporal_model = train_temporal(
            temporal_model, encoders, train_dl, val_dl, cfg, device,
        )
    else:
        ckpt = Path(cfg["project"]["checkpoint_dir"]) / "temporal" / "best.pt"
        if ckpt.exists():
            temporal_model.load_state_dict(torch.load(ckpt, weights_only=True))
            temporal_model.to(device)

    # ── Stage 3 ─────────────────────────────────────────────────────
    if start <= 3:
        logger.info("═══ STAGE 3: Training Reverse Generator ═══")
        decoder = train_decoder(
            decoder, temporal_model, encoders, train_dl, val_dl, cfg, device,
        )
    else:
        ckpt = Path(cfg["project"]["checkpoint_dir"]) / "decoder" / "best.pt"
        if ckpt.exists():
            decoder.load_state_dict(torch.load(ckpt, weights_only=True))
            decoder.to(device)

    # ── Stage 4 ─────────────────────────────────────────────────────
    if start <= 4:
        logger.info("═══ STAGE 4: Training Fusion Model ═══")
        fusion_model = train_fusion(
            fusion_model, decoder, temporal_model, encoders,
            train_dl, val_dl, cfg, device,
        )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()