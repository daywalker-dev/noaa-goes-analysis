"""
Streaming Download + Preprocess Pipeline
==========================================
Downloads one timestep at a time, preprocesses immediately, appends to Zarr,
then deletes the raw NetCDF file. This keeps disk usage under a configurable
budget (default 10 GB).

Usage:
    python main.py stream --config config/default.yaml
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default disk budget in bytes (10 GB)
DEFAULT_DISK_BUDGET_BYTES = 10 * 1024 * 1024 * 1024

# Default download interval in hours (3h instead of 1h to reduce volume)
DEFAULT_INTERVAL_HOURS = 3


class StreamingPipeline:
    """Download → preprocess → append to Zarr → delete raw, one timestep at a time.

    Config keys used:
        data.satellite          — e.g. "goes16"
        data.products           — list of product dicts with id/domain/channels
        data.raw_dir            — temp directory for raw downloads (cleaned after each step)
        data.zarr_path          — output Zarr store path
        data.stats_path         — output normalization stats JSON
        data.date_range.start   — start date string
        data.date_range.end     — end date string
        data.streaming.interval_hours  — hours between downloads (default 3)
        data.streaming.disk_budget_gb  — max disk usage in GB (default 10)
        data.streaming.goes2go_cache   — path to goes2go cache to clean (default ~/data)
        data.preprocessing.*    — preprocessing parameters
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.satellite = cfg.data.satellite
        self.raw_dir = Path(cfg.data.raw_dir)
        self.zarr_path = Path(cfg.data.get("zarr_path", "data/processed/goes_l2.zarr"))
        self.stats_path = Path(cfg.data.get("stats_path", "data/processed/stats.json"))

        # Streaming config with defaults
        streaming = cfg.data.get("streaming", {})
        self.interval_hours = streaming.get("interval_hours", DEFAULT_INTERVAL_HOURS)
        self.disk_budget_bytes = streaming.get("disk_budget_gb", 10) * 1024 * 1024 * 1024
        self.goes2go_cache = Path(streaming.get("goes2go_cache", Path.home() / "data"))

        # Parse date range
        dr = cfg.data.get("date_range", {})
        self.start = datetime.fromisoformat(dr.get("start", "2023-06-01"))
        self.end = datetime.fromisoformat(dr.get("end", "2023-08-31"))

        # Build product info
        self.products = []
        self.channel_names = []
        for p in cfg.data.products:
            pid = p["id"] if isinstance(p, dict) else p
            shortname = pid.split("-")[-1].replace("F", "").replace("P", "")
            channels = p.get("channels", [shortname]) if isinstance(p, dict) else [shortname]
            self.products.append({
                "id": pid,
                "shortname": shortname,
                "domain": p.get("domain", "unknown") if isinstance(p, dict) else "unknown",
                "channels": channels,
                "required": p.get("required", True) if isinstance(p, dict) else True,
            })
            self.channel_names.extend(channels)

        self.total_channels = len(self.channel_names)

        # Import preprocessor lazily
        from data.preprocessor import GOESPreprocessor
        self.preprocessor = GOESPreprocessor(cfg)

    def run(self) -> None:
        """Execute the full streaming pipeline."""
        logger.info(
            f"Streaming pipeline: {self.start} → {self.end}, "
            f"interval={self.interval_hours}h, budget={self.disk_budget_bytes / 1e9:.1f}GB"
        )

        # Initialize Zarr store (or open existing)
        zarr_store = self._init_zarr()

        # Generate timestep sequence
        current = self.start
        timestep_count = 0
        all_processed = []
        running_stats = {name: {"sum": 0.0, "sum_sq": 0.0, "count": 0}
                         for name in self.channel_names}

        while current <= self.end:
            # Check disk budget before downloading
            self._check_disk_budget()

            logger.info(f"Processing timestep: {current.isoformat()}")

            try:
                # Download one timestep for all products
                raw_files = self._download_timestep(current)

                if raw_files:
                    # Preprocess all products for this timestep
                    frame_data = self._preprocess_timestep(raw_files)

                    if frame_data is not None:
                        # Append to Zarr
                        self._append_to_zarr(zarr_store, frame_data, current)
                        timestep_count += 1

                        # Update running stats
                        self._update_running_stats(running_stats, frame_data)

                # Delete raw files immediately
                self._cleanup_raw_files(raw_files)

                # Also clean goes2go cache
                self._clean_goes2go_cache()

            except Exception as e:
                logger.warning(f"Failed to process {current}: {e}")
                # Still clean up on failure
                self._cleanup_raw_files({})
                self._clean_goes2go_cache()

            current += timedelta(hours=self.interval_hours)

        # Finalize: write normalization stats
        if timestep_count > 0:
            stats = self._finalize_stats(running_stats)
            self._write_stats(stats)
            logger.info(
                f"Pipeline complete: {timestep_count} timesteps written to {self.zarr_path}"
            )
        else:
            logger.error("No timesteps were successfully processed")

    def _download_timestep(self, timestamp: datetime) -> dict[str, Path]:
        """Download all products for a single timestep.

        Returns:
            Dict mapping product shortname → downloaded file path.
        """
        from goes2go import GOES

        raw_files = {}
        for prod in self.products:
            product_id = prod["id"]
            shortname = prod["shortname"]

            try:
                G = GOES(satellite=self.satellite, product=product_id, domain="F")
                # Download a single observation nearest to the timestamp
                ds = G.nearesttime(timestamp, within=timedelta(minutes=90))

                if ds is not None:
                    # goes2go typically saves files — find the most recent .nc
                    cache_dir = self.goes2go_cache / self.satellite
                    nc_files = sorted(cache_dir.rglob("*.nc"), key=lambda p: p.stat().st_mtime)
                    if nc_files:
                        latest = nc_files[-1]
                        # Copy to our raw_dir so we control cleanup
                        dest_dir = self.raw_dir / shortname
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / latest.name
                        shutil.copy2(latest, dest)
                        raw_files[shortname] = dest
                        logger.debug(f"  Downloaded {shortname}: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")

            except Exception as e:
                if prod["required"]:
                    logger.warning(f"  Failed to download {shortname}: {e}")
                else:
                    logger.debug(f"  Optional product {shortname} unavailable: {e}")

        return raw_files

    def _preprocess_timestep(self, raw_files: dict[str, Path]) -> Optional[np.ndarray]:
        """Preprocess raw files for one timestep into a single (C, H, W) array.

        Returns:
            np.ndarray of shape (C, H, W) or None if preprocessing failed.
        """
        all_channels = []

        for prod in self.products:
            shortname = prod["shortname"]
            if shortname not in raw_files:
                if prod["required"]:
                    # Fill with NaN for required missing products
                    n_ch = len(prod["channels"])
                    h, w = self.preprocessor.grid_shape
                    all_channels.append(np.full((n_ch, h, w), np.nan, dtype=np.float32))
                continue

            fpath = raw_files[shortname]
            results = self.preprocessor.process_files([fpath], prod["id"])

            if results:
                all_channels.append(results[0]["data"])  # (C_prod, H, W)
            elif prod["required"]:
                n_ch = len(prod["channels"])
                h, w = self.preprocessor.grid_shape
                all_channels.append(np.full((n_ch, h, w), np.nan, dtype=np.float32))

        if not all_channels:
            return None

        # Concatenate along channel dimension
        return np.concatenate(all_channels, axis=0)  # (C_total, H, W)

    def _init_zarr(self):
        """Initialize or open the output Zarr store."""
        import zarr
        from numcodecs import Blosc

        self.zarr_path.parent.mkdir(parents=True, exist_ok=True)

        if self.zarr_path.exists():
            store = zarr.open(str(self.zarr_path), mode="r+")
            logger.info(f"Opened existing Zarr store: {store['data'].shape}")
            return store

        compressor = Blosc(cname="lz4", clevel=5, shuffle=2)  # BITSHUFFLE
        h, w = self.preprocessor.grid_shape

        store = zarr.open(str(self.zarr_path), mode="w")
        store.create_dataset(
            "data",
            shape=(0, self.total_channels, h, w),
            chunks=(1, self.total_channels, h, w),
            dtype="float32",
            compressor=compressor,
            # Allow appending along time axis
        )
        store.create_dataset(
            "times",
            shape=(0,),
            chunks=(1024,),
            dtype="int64",
        )
        store.create_dataset("lats", data=self.preprocessor.target_lats)
        store.create_dataset("lons", data=self.preprocessor.target_lons)
        store.attrs["channel_names"] = self.channel_names

        logger.info(f"Created new Zarr store: C={self.total_channels}, H={h}, W={w}")
        return store

    def _append_to_zarr(self, store, frame_data: np.ndarray, timestamp: datetime) -> None:
        """Append a single frame (C, H, W) to the Zarr store."""
        data_arr = store["data"]
        times_arr = store["times"]

        current_t = data_arr.shape[0]

        # Resize and append
        data_arr.resize(current_t + 1, *data_arr.shape[1:])
        data_arr[current_t] = frame_data.astype(np.float32)

        ts_ns = np.datetime64(timestamp, "ns").astype(np.int64)
        times_arr.resize(current_t + 1)
        times_arr[current_t] = ts_ns

    def _update_running_stats(self, stats: dict, frame_data: np.ndarray) -> None:
        """Update running mean/variance statistics (Welford's online algorithm)."""
        for i, name in enumerate(self.channel_names):
            if i >= frame_data.shape[0]:
                break
            channel = frame_data[i]
            valid = channel[np.isfinite(channel)]
            if len(valid) > 0:
                stats[name]["sum"] += float(np.sum(valid))
                stats[name]["sum_sq"] += float(np.sum(valid ** 2))
                stats[name]["count"] += len(valid)

    def _finalize_stats(self, running_stats: dict) -> dict:
        """Compute final mean/std from running accumulators."""
        stats = {}
        for name, s in running_stats.items():
            n = s["count"]
            if n > 0:
                mean = s["sum"] / n
                var = (s["sum_sq"] / n) - mean ** 2
                std = float(np.sqrt(max(var, 0)))
                stats[name] = {"mean": float(mean), "std": std if std > 1e-8 else 1.0}
            else:
                stats[name] = {"mean": 0.0, "std": 1.0}
        return stats

    def _write_stats(self, stats: dict) -> None:
        """Write normalization statistics to JSON."""
        import json
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats written to {self.stats_path}")

    def _cleanup_raw_files(self, raw_files: dict[str, Path]) -> None:
        """Delete all raw downloaded files."""
        # Delete specific files
        for shortname, fpath in raw_files.items():
            try:
                if fpath.exists():
                    fpath.unlink()
                    logger.debug(f"  Deleted raw file: {fpath}")
            except OSError as e:
                logger.warning(f"  Failed to delete {fpath}: {e}")

        # Also clean any remaining .nc files in raw_dir
        if self.raw_dir.exists():
            for nc in self.raw_dir.rglob("*.nc"):
                try:
                    nc.unlink()
                except OSError:
                    pass

    def _clean_goes2go_cache(self) -> None:
        """Remove goes2go's download cache to reclaim disk space.

        goes2go stores files in ~/data/<satellite>/ by default.
        We aggressively clean this after each timestep.
        """
        cache_dir = self.goes2go_cache
        if not cache_dir.exists():
            return

        total_freed = 0
        for nc in cache_dir.rglob("*.nc"):
            try:
                size = nc.stat().st_size
                nc.unlink()
                total_freed += size
            except OSError:
                pass

        if total_freed > 0:
            logger.debug(f"  Cleaned goes2go cache: freed {total_freed / 1e6:.1f} MB")

    def _check_disk_budget(self) -> None:
        """Check total disk usage and warn/pause if approaching budget."""
        total_usage = 0

        # Check Zarr store size
        if self.zarr_path.exists():
            total_usage += _dir_size(self.zarr_path)

        # Check raw dir
        if self.raw_dir.exists():
            total_usage += _dir_size(self.raw_dir)

        # Check goes2go cache
        if self.goes2go_cache.exists():
            total_usage += _dir_size(self.goes2go_cache)

        usage_gb = total_usage / 1e9
        budget_gb = self.disk_budget_bytes / 1e9

        if total_usage > self.disk_budget_bytes * 0.9:
            logger.warning(
                f"Disk usage {usage_gb:.1f}GB approaching budget {budget_gb:.1f}GB — "
                "cleaning caches aggressively"
            )
            self._clean_goes2go_cache()
            self._cleanup_raw_files({})

        if total_usage > self.disk_budget_bytes:
            logger.error(
                f"Disk usage {usage_gb:.1f}GB exceeds budget {budget_gb:.1f}GB — "
                "stopping pipeline"
            )
            raise RuntimeError(f"Disk budget exceeded: {usage_gb:.1f}GB > {budget_gb:.1f}GB")


def _dir_size(path: Path) -> int:
    """Compute total size of a directory tree in bytes."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except OSError:
        pass
    return total