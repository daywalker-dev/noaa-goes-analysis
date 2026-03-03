"""Sliding-window PyTorch Dataset over the GOES L2 Zarr store."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig

from goes_forecast.data.zarr_store import ZarrStore
from goes_forecast.data.augmentation import build_augmentation

logger = logging.getLogger(__name__)


class GOESWindowDataset(Dataset):
    """Sliding window dataset serving (input, target) pairs from Zarr.

    Each item returns:
        input_data:  (T_in, C, pH, pW)  — input context window
        target_data: (T_out, C, pH, pW) — forecast target window
        mask:        (T_out, 1, pH, pW) — validity mask for target
        meteo_input: (T_in, F)          — meteorological fields (spatial avg)
        meteo_target:(T_out, F)         — meteorological target fields
    """

    def __init__(
        self,
        zarr_path: str | Path,
        stats_path: Optional[str | Path],
        cfg: DictConfig,
        split: str = "train",
        transform=None,
    ):
        self.store = ZarrStore(zarr_path, stats_path)
        self.cfg = cfg
        self.split = split
        self.transform = transform

        self.input_steps = cfg.data.temporal.input_steps
        self.forecast_steps = cfg.data.temporal.forecast_steps
        self.total_steps = self.input_steps + self.forecast_steps
        self.patch_size = cfg.data.spatial.patch_size
        self.patch_stride = cfg.data.spatial.patch_stride

        # Get normalization stats
        self.means, self.stds = self.store.get_stats_arrays()

        # Determine domain channel indices
        products = cfg.data.products
        self.domain_indices = {}
        for domain in ["land", "sea", "cloud", "meteo"]:
            self.domain_indices[domain] = self.store.get_channel_indices(
                domain, [dict(p) for p in products]
            )

        # Build valid window indices
        self._build_indices()

    def _build_indices(self) -> None:
        """Build all valid (time, lat, lon) start indices for the split."""
        split_range = self.cfg.data.splits[self.split]
        t_start = self.store.time_index_for_date(split_range[0])
        t_end = self.store.time_index_for_date(split_range[1])

        # Valid time starts: need total_steps consecutive hours
        max_t = min(t_end, self.store.T - self.total_steps)
        time_indices = list(range(t_start, max_t + 1))

        # Spatial patch starts
        H, W = self.store.H, self.store.W
        lat_starts = list(range(0, H - self.patch_size + 1, self.patch_stride))
        lon_starts = list(range(0, W - self.patch_size + 1, self.patch_stride))

        if not lat_starts:
            lat_starts = [0]
        if not lon_starts:
            lon_starts = [0]

        self.indices = [
            (t, lat, lon)
            for t in time_indices
            for lat in lat_starts
            for lon in lon_starts
        ]
        logger.info(
            f"Dataset [{self.split}]: {len(self.indices)} samples "
            f"({len(time_indices)} times × {len(lat_starts)} lat × {len(lon_starts)} lon)"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t, lat, lon = self.indices[idx]
        ps = self.patch_size
        lat_end = min(lat + ps, self.store.H)
        lon_end = min(lon + ps, self.store.W)

        # Read spatio-temporal patch: (total_steps, C, pH, pW)
        patch = self.store.get_patch(t, t + self.total_steps, lat, lat_end, lon, lon_end)
        patch = patch.astype(np.float32)

        # Replace NaN with 0
        mask = np.isfinite(patch).all(axis=1, keepdims=True).astype(np.float32)
        patch = np.nan_to_num(patch, nan=0.0)

        # Normalize
        patch = (patch - self.means[None]) / np.where(self.stds < 1e-8, 1.0, self.stds)[None]

        # Pad if patch is smaller than expected
        if patch.shape[2] < ps or patch.shape[3] < ps:
            padded = np.zeros((self.total_steps, self.store.C, ps, ps), dtype=np.float32)
            padded[:, :, :patch.shape[2], :patch.shape[3]] = patch
            patch = padded
            mask_padded = np.zeros((self.total_steps, 1, ps, ps), dtype=np.float32)
            mask_padded[:, :, :mask.shape[2], :mask.shape[3]] = mask
            mask = mask_padded

        # Split into input and target
        input_data = torch.from_numpy(patch[:self.input_steps])
        target_data = torch.from_numpy(patch[self.input_steps:])
        target_mask = torch.from_numpy(mask[self.input_steps:])

        # Extract meteorological fields (spatial average per timestep)
        meteo_idx = self.domain_indices.get("meteo", [])
        if meteo_idx:
            meteo_in = input_data[:, meteo_idx].mean(dim=(-2, -1))  # (T_in, F)
            meteo_tgt = target_data[:, meteo_idx].mean(dim=(-2, -1))  # (T_out, F)
        else:
            meteo_in = torch.zeros(self.input_steps, 1)
            meteo_tgt = torch.zeros(self.forecast_steps, 1)

        # Apply augmentation
        if self.transform and self.split == "train":
            for t_step in range(input_data.shape[0]):
                input_data[t_step] = self.transform(input_data[t_step])
            for t_step in range(target_data.shape[0]):
                target_data[t_step] = self.transform(target_data[t_step])

        return {
            "input": input_data,          # (T_in, C, H, W)
            "target": target_data,        # (T_out, C, H, W)
            "mask": target_mask,          # (T_out, 1, H, W)
            "meteo_input": meteo_in,      # (T_in, F)
            "meteo_target": meteo_tgt,    # (T_out, F)
            "domain_indices": {k: torch.tensor(v) for k, v in self.domain_indices.items()},
        }


def build_dataloader(
    cfg: DictConfig,
    split: str = "train",
) -> DataLoader:
    """Factory to build a DataLoader for a given split."""
    transform = build_augmentation(cfg) if split == "train" else None
    dataset = GOESWindowDataset(
        zarr_path=cfg.data.zarr_path,
        stats_path=cfg.data.stats_path,
        cfg=cfg,
        split=split,
        transform=transform,
    )
    loader_cfg = cfg.data.loader
    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        prefetch_factor=loader_cfg.get("prefetch_factor", 2),
        drop_last=(split == "train"),
    )
