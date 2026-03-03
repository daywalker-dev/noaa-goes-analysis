"""GOES L2 product downloader using goes2go."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GOESDownloader:
    """Fetches GOES L2 products for a date range using goes2go.

    Args:
        satellite: Satellite name (e.g. 'GOES-16', 'GOES-17', 'GOES-18').
        output_dir: Base directory for downloaded NetCDF files.
        products: List of ABI L2 product IDs to download.
    """

    PRODUCT_MAP = {
        "SST": "ABI-L2-SSTF",
        "LST": "ABI-L2-LSTF",
        "CMIP": "ABI-L2-CMIPF",
        "DMW": "ABI-L2-DMWF",
        "TPW": "ABI-L2-TPWF",
        "AOD": "ABI-L2-AODF",
    }

    def __init__(
        self,
        satellite: str = "goes16",
        output_dir: str | Path = "data/raw",
        products: Optional[list[str]] = None,
    ):
        self.satellite = satellite.lower().replace("-", "")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.products = products or list(self.PRODUCT_MAP.keys())

    def _resolve_product(self, shortname: str) -> str:
        return self.PRODUCT_MAP.get(shortname.upper(), shortname)

    def download(
        self,
        start: str | datetime,
        end: str | datetime,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Download all configured products for the given date range.

        Returns:
            DataFrame manifest with columns [product, path, timestamp, size_mb].
        """
        from goes2go import GOES

        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)

        records = []
        for shortname in self.products:
            product_id = self._resolve_product(shortname)
            product_dir = self.output_dir / shortname
            product_dir.mkdir(exist_ok=True)

            logger.info(f"Downloading {product_id} from {start} to {end}")

            for attempt in range(max_retries):
                try:
                    G = GOES(satellite=self.satellite, product=product_id, domain="F")
                    ds = G.timerange(start=start, end=end, return_as="filelist")

                    if ds is not None and len(ds) > 0:
                        for _, row in ds.iterrows():
                            fpath = Path(row.get("file", ""))
                            ts = row.get("start", start)
                            size = fpath.stat().st_size / 1e6 if fpath.exists() else 0
                            records.append({
                                "product": shortname,
                                "path": str(fpath),
                                "timestamp": ts,
                                "size_mb": round(size, 2),
                            })
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    logger.warning(f"Attempt {attempt+1} failed for {product_id}: {e}. Retrying in {wait}s")
                    time.sleep(wait)
            else:
                logger.error(f"Failed to download {product_id} after {max_retries} attempts")

        manifest = pd.DataFrame(records)
        manifest_path = self.output_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        logger.info(f"Download manifest saved to {manifest_path} ({len(manifest)} files)")
        return manifest

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
