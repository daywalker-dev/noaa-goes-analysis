"""GOES L2 product downloader using goes2go — disk-aware version.

Changes from original:
    - download() now accepts interval_hours to skip intermediate timesteps
    - download_single() method for one-at-a-time operation
    - Automatic cleanup of goes2go cache after each download
    - Disk budget checking before each download
    - cleanup_cache() public method
"""
from __future__ import annotations

import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DISK_BUDGET_GB = 10


class GOESDownloader:
    """Fetches GOES L2 products for a date range using goes2go.

    Args:
        satellite: Satellite name (e.g. 'goes16', 'goes17', 'goes18').
        output_dir: Base directory for downloaded NetCDF files.
        products: List of ABI L2 product short names to download.
    """

    PRODUCT_MAP = {
        "SST": "ABI-L2-SSTF",
        "LST": "ABI-L2-LSTF",
        "CMIP": "ABI-L2-CMIPF",
        "CMI": "ABI-L2-CMIPF",
        "DMW": "ABI-L2-DMWF",
        "TPW": "ABI-L2-TPWF",
        "AOD": "ABI-L2-AODF",
        "ADP": "ABI-L2-ADPF",
    }

    def __init__(
        self,
        satellite: str = "goes16",
        output_dir: str | Path = "data/raw",
        products: Optional[list[str]] = None,
        disk_budget_gb: float = DEFAULT_DISK_BUDGET_GB,
    ):
        self.satellite = satellite.lower().replace("-", "")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.products = products or list(self.PRODUCT_MAP.keys())
        self.disk_budget_bytes = disk_budget_gb * 1024 * 1024 * 1024
        self._goes2go_cache = Path.home() / "data"

    def _resolve_product(self, shortname: str) -> str:
        return self.PRODUCT_MAP.get(shortname.upper(), shortname)

    def download(
        self,
        start: str | datetime,
        end: str | datetime,
        max_retries: int = 3,
        interval_hours: int = 3,
        cleanup_after_each: bool = True,
    ) -> pd.DataFrame:
        """Download all configured products for the given date range.

        Args:
            start: Start date/datetime.
            end: End date/datetime.
            max_retries: Max retry attempts per product per timestep.
            interval_hours: Hours between downloads (default 3 to limit volume).
            cleanup_after_each: If True, delete raw files and goes2go cache
                after preprocessing each timestep.

        Returns:
            DataFrame manifest with columns [product, path, timestamp, size_mb].
        """
        from goes2go import GOES

        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        records = []
        current = start

        while current <= end:
            # Check disk budget
            if self._check_disk_usage() > self.disk_budget_bytes * 0.9:
                logger.warning("Approaching disk budget — cleaning caches")
                self.cleanup_cache()

            for shortname in self.products:
                product_id = self._resolve_product(shortname)
                product_dir = self.output_dir / shortname
                product_dir.mkdir(exist_ok=True)

                for attempt in range(max_retries):
                    try:
                        G = GOES(satellite=self.satellite, product=product_id, domain="F")
                        ds = G.nearesttime(current, within=timedelta(minutes=90))

                        if ds is not None:
                            # Find the downloaded file from goes2go cache
                            fpath = self._find_latest_nc()
                            if fpath:
                                dest = product_dir / fpath.name
                                shutil.copy2(fpath, dest)
                                size = dest.stat().st_size / 1e6
                                records.append({
                                    "product": shortname,
                                    "path": str(dest),
                                    "timestamp": current,
                                    "size_mb": round(size, 2),
                                })
                                logger.debug(f"  {shortname} @ {current}: {size:.1f} MB")
                        break

                    except Exception as e:
                        wait = 2 ** attempt
                        logger.warning(
                            f"Attempt {attempt+1} failed for {product_id} @ {current}: {e}. "
                            f"Retrying in {wait}s"
                        )
                        time.sleep(wait)
                else:
                    logger.error(f"Failed to download {product_id} @ {current} after {max_retries} attempts")

            # Clean goes2go cache after each timestep to limit disk growth
            if cleanup_after_each:
                self.cleanup_cache()

            current += timedelta(hours=interval_hours)

        manifest = pd.DataFrame(records)
        if not manifest.empty:
            manifest_path = self.output_dir / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)
            logger.info(f"Download manifest saved to {manifest_path} ({len(manifest)} files)")
        return manifest

    def download_single(
        self,
        timestamp: datetime,
        product_shortname: str,
        max_retries: int = 3,
    ) -> Optional[Path]:
        """Download a single product for a single timestamp.

        Returns:
            Path to downloaded file, or None if failed.
        """
        from goes2go import GOES

        product_id = self._resolve_product(product_shortname)
        product_dir = self.output_dir / product_shortname
        product_dir.mkdir(exist_ok=True)

        for attempt in range(max_retries):
            try:
                G = GOES(satellite=self.satellite, product=product_id, domain="F")
                ds = G.nearesttime(timestamp, within=timedelta(minutes=90))

                if ds is not None:
                    fpath = self._find_latest_nc()
                    if fpath:
                        dest = product_dir / fpath.name
                        shutil.copy2(fpath, dest)
                        return dest
                return None

            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)

        return None

    def cleanup_cache(self) -> int:
        """Remove goes2go cache files. Returns bytes freed."""
        freed = 0
        if self._goes2go_cache.exists():
            for nc in self._goes2go_cache.rglob("*.nc"):
                try:
                    freed += nc.stat().st_size
                    nc.unlink()
                except OSError:
                    pass
        if freed > 0:
            logger.debug(f"Cleaned goes2go cache: freed {freed / 1e6:.1f} MB")
        return freed

    def cleanup_raw(self) -> int:
        """Remove all raw NetCDF files from output_dir. Returns bytes freed."""
        freed = 0
        for nc in self.output_dir.rglob("*.nc"):
            try:
                freed += nc.stat().st_size
                nc.unlink()
            except OSError:
                pass
        if freed > 0:
            logger.debug(f"Cleaned raw files: freed {freed / 1e6:.1f} MB")
        return freed

    def build_manifest(self) -> pd.DataFrame:
        """Scan output_dir for existing files and build a manifest."""
        records = []
        for nc_file in self.output_dir.rglob("*.nc"):
            product = nc_file.parent.name
            records.append({
                "product": product,
                "path": str(nc_file),
                "timestamp": None,
                "size_mb": round(nc_file.stat().st_size / 1e6, 2),
            })
        return pd.DataFrame(records)

    def _find_latest_nc(self) -> Optional[Path]:
        """Find the most recently modified .nc file in the goes2go cache."""
        if not self._goes2go_cache.exists():
            return None
        nc_files = sorted(self._goes2go_cache.rglob("*.nc"), key=lambda p: p.stat().st_mtime)
        return nc_files[-1] if nc_files else None

    def _check_disk_usage(self) -> int:
        """Return total bytes used by output_dir + goes2go cache."""
        total = 0
        for base in [self.output_dir, self._goes2go_cache]:
            if base.exists():
                for f in base.rglob("*"):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except OSError:
                            pass
        return total