"""Zarr store read/write and chunking utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ZarrStore:
    """Lightweight wrapper around a Zarr store for GOES L2 data.

    Expected store layout:
        /data   : float32 (T, C, H, W)
        /times  : int64 (T,) — nanosecond timestamps
        /lats   : float64 (H,)
        /lons   : float64 (W,)
        attrs.channel_names : list[str]
    """

    def __init__(self, zarr_path: str | Path, stats_path: Optional[str | Path] = None):
        import zarr
        self.path = Path(zarr_path)
        self.store = zarr.open(str(self.path), mode="r")
        self.data = self.store["data"]
        self.times = np.array(self.store["times"]).view("datetime64[ns]")
        self.lats = np.array(self.store["lats"])
        self.lons = np.array(self.store["lons"])
        self.channel_names = list(self.store.attrs.get("channel_names", []))
        self.T, self.C, self.H, self.W = self.data.shape

        self.stats = {}
        if stats_path and Path(stats_path).exists():
            with open(stats_path) as f:
                self.stats = json.load(f)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.T, self.C, self.H, self.W)

    def get_window(self, t_start: int, t_end: int) -> np.ndarray:
        """Read a temporal window. Returns (t_end - t_start, C, H, W)."""
        return np.array(self.data[t_start:t_end])

    def get_patch(
        self,
        t_start: int,
        t_end: int,
        lat_start: int,
        lat_end: int,
        lon_start: int,
        lon_end: int,
    ) -> np.ndarray:
        """Read a spatio-temporal patch. Returns (T, C, pH, pW)."""
        return np.array(self.data[t_start:t_end, :, lat_start:lat_end, lon_start:lon_end])

    def get_channel_indices(self, domain: str, products: list[dict]) -> list[int]:
        """Get channel indices for a specific domain from product config."""
        indices = []
        offset = 0
        for prod in products:
            n_ch = len(prod["channels"])
            if prod["domain"] == domain:
                indices.extend(range(offset, offset + n_ch))
            offset += n_ch
        return indices

    def get_stats_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get (means, stds) arrays shaped (C, 1, 1) for normalization."""
        means = np.zeros(self.C, dtype=np.float32)
        stds = np.ones(self.C, dtype=np.float32)
        for i, name in enumerate(self.channel_names):
            if name in self.stats:
                means[i] = self.stats[name]["mean"]
                stds[i] = self.stats[name]["std"]
        return means[:, None, None], stds[:, None, None]

    def time_index_for_date(self, date_str: str) -> int:
        """Find the time index closest to a date string."""
        target = np.datetime64(date_str)
        idx = np.argmin(np.abs(self.times - target))
        return int(idx)
