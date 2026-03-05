"""Preprocess GOES L2 NetCDF files → aligned, normalized Zarr store."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from omegaconf import DictConfig
from scipy.ndimage import distance_transform_edt

from utils.projection import (
    compute_grid_coords,
    goes_fixed_grid_to_latlon,
    make_target_area,
    reproject_to_grid,
)

logger = logging.getLogger(__name__)


class GOESPreprocessor:
    """Preprocess raw GOES L2 NetCDF files into a unified Zarr store.

    Pipeline:
        1. Parse raw files, extract data + DQF
        2. Reproject to common lat/lon grid
        3. Apply quality filtering
        4. Temporal alignment to hourly axis
        5. Compute normalization statistics (train split only)
        6. Normalize and write to Zarr
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        prep = cfg.data.preprocessing
        self.lat_range = tuple(prep.lat_range)
        self.lon_range = tuple(prep.lon_range)
        self.resolution = prep.target_resolution_deg
        self.quality_threshold = prep.quality_threshold
        self.fill_strategy = prep.fill_strategy
        self.gap_fill_minutes = prep.gap_fill_minutes

        self.target_lats, self.target_lons = compute_grid_coords(
            self.lat_range, self.lon_range, self.resolution
        )
        self.grid_shape = (len(self.target_lats), len(self.target_lons))

    def _parse_goes_file(self, path: str | Path) -> Optional[dict]:
        """Parse a GOES NetCDF file and extract data + coordinates."""
        path = Path(path)
        try:
            ds = xr.open_dataset(path, engine="netcdf4")
        except Exception as e:
            logger.warning(f"Failed to open {path}: {e}")
            return None

        # Extract projection info if available
        proj_info = {}
        if "goes_imager_projection" in ds:
            gip = ds["goes_imager_projection"]
            proj_info = {
                "perspective_point_height": float(gip.attrs.get("perspective_point_height", 35786023)),
                "semi_major_axis": float(gip.attrs.get("semi_major_axis", 6378137.0)),
                "semi_minor_axis": float(gip.attrs.get("semi_minor_axis", 6356752.31414)),
                "longitude_of_projection_origin": float(gip.attrs.get("longitude_of_projection_origin", -75.0)),
            }

        # Extract data variables (skip coords and projection)
        skip = {"goes_imager_projection", "x", "y", "t", "time", "time_bounds"}
        data_vars = {}
        for name in ds.data_vars:
            if name in skip or name.startswith("nominal"):
                continue
            arr = ds[name].values
            if arr.ndim >= 2:
                data_vars[name] = arr.astype(np.float32)

        # Get DQF if present
        dqf = data_vars.pop("DQF", None)

        result = {
            "data_vars": data_vars,
            "dqf": dqf,
            "proj_info": proj_info,
        }

        # Extract x, y coordinates
        if "x" in ds and "y" in ds:
            result["x"] = ds["x"].values
            result["y"] = ds["y"].values

        ds.close()
        return result

    def _compute_quality_mask(self, dqf: Optional[np.ndarray]) -> np.ndarray:
        """Compute binary quality mask from DQF. 1=valid, 0=bad."""
        if dqf is None:
            return np.ones(self.grid_shape, dtype=np.float32)
        valid = (dqf == 0).astype(np.float32)
        valid_fraction = np.nanmean(valid)
        if valid_fraction < self.quality_threshold:
            return np.zeros(self.grid_shape, dtype=np.float32)
        return valid

    def _fill_missing(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fill missing values based on configured strategy."""
        if self.fill_strategy == "zero":
            return np.where(mask > 0, data, 0.0)
        elif self.fill_strategy == "mask":
            return np.where(mask > 0, data, np.nan)
        else:  # interpolate via nearest valid pixel
            if np.all(mask > 0):
                return data
            invalid = mask == 0
            if np.all(invalid):
                return np.zeros_like(data)
            _, indices = distance_transform_edt(invalid, return_distances=True, return_indices=True)
            return data[tuple(indices)]

    def _reproject_scene(
        self,
        data: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        proj_info: dict,
    ) -> np.ndarray:
        """Reproject a single 2D scene from GOES fixed-grid to lat/lon."""
        try:
            target_area = make_target_area(self.lat_range, self.lon_range, self.resolution)
            src_lats, src_lons = goes_fixed_grid_to_latlon(x, y, proj_info)
            return reproject_to_grid(data, src_lats, src_lons, target_area)
        except ImportError:
            # Fallback: simple scipy interpolation
            from scipy.interpolate import RegularGridInterpolator
            src_y = np.linspace(self.lat_range[0], self.lat_range[1], data.shape[0])
            src_x = np.linspace(self.lon_range[0], self.lon_range[1], data.shape[1])
            interp = RegularGridInterpolator(
                (src_y, src_x), data, method="linear",
                bounds_error=False, fill_value=np.nan,
            )
            grid_lat, grid_lon = np.meshgrid(self.target_lats, self.target_lons, indexing="ij")
            return interp((grid_lat, grid_lon)).astype(np.float32)

    def process_files(
        self,
        file_list: list[str | Path],
        product_id: str,
    ) -> list[dict]:
        """Process a list of files for a single product.

        Returns:
            List of dicts with 'data' (C, H, W), 'mask' (H, W), 'timestamp'.
        """
        results = []
        for fpath in file_list:
            parsed = self._parse_goes_file(fpath)
            if parsed is None:
                continue

            mask = self._compute_quality_mask(parsed.get("dqf"))
            if np.mean(mask) < self.quality_threshold:
                continue

            channels = []
            for name, arr in parsed["data_vars"].items():
                if arr.ndim == 2:
                    if "x" in parsed and "proj_info" in parsed:
                        reproj = self._reproject_scene(
                            arr, parsed["x"], parsed["y"], parsed["proj_info"]
                        )
                    else:
                        reproj = self._resize_to_grid(arr)
                    filled = self._fill_missing(reproj, mask)
                    channels.append(filled)

            if channels:
                data = np.stack(channels, axis=0)  # (C, H, W)
                results.append({"data": data, "mask": mask, "product": product_id})

        logger.info(f"Processed {len(results)}/{len(file_list)} scenes for {product_id}")
        return results

    def _resize_to_grid(self, data: np.ndarray) -> np.ndarray:
        """Resize data to target grid shape using simple interpolation."""
        from scipy.ndimage import zoom
        factors = (self.grid_shape[0] / data.shape[0], self.grid_shape[1] / data.shape[1])
        return zoom(data, factors, order=1).astype(np.float32)

    def compute_stats(self, data_dict: dict[str, np.ndarray]) -> dict[str, dict]:
        """Compute per-channel mean and std from training data.

        Args:
            data_dict: {channel_name: array of shape (N, H, W)}

        Returns:
            {channel_name: {"mean": float, "std": float}}
        """
        stats = {}
        for name, arr in data_dict.items():
            valid = arr[~np.isnan(arr)]
            stats[name] = {
                "mean": float(np.mean(valid)) if len(valid) > 0 else 0.0,
                "std": float(np.std(valid)) if len(valid) > 0 else 1.0,
            }
        return stats

    def normalize(self, data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Apply z-score normalization. Shapes: data (C,H,W), means/stds (C,1,1)."""
        stds = np.where(stds < 1e-8, 1.0, stds)
        return (data - means) / stds

    def write_zarr(
        self,
        output_path: str | Path,
        data: np.ndarray,
        channel_names: list[str],
        times: np.ndarray,
        stats: dict,
    ) -> None:
        """Write preprocessed data to chunked Zarr store.

        Args:
            output_path: Path for the Zarr store.
            data: Array of shape (T, C, H, W).
            channel_names: List of channel names.
            times: 1D array of timestamps.
            stats: Normalization statistics dict.
        """
        import zarr
        from numcodecs import Blosc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

        store = zarr.open(str(output_path), mode="w")
        T, C, H, W = data.shape

        store.create_dataset(
            "data", shape=(T, C, H, W), chunks=(1, C, H, W),
            dtype="float32", compressor=compressor,
        )
        store["data"][:] = data

        store.create_dataset("times", data=times.astype("datetime64[ns]").astype(np.int64))
        store.create_dataset("lats", data=self.target_lats)
        store.create_dataset("lons", data=self.target_lons)
        store.attrs["channel_names"] = channel_names

        # Save stats alongside
        stats_path = Path(self.cfg.data.stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Zarr store written to {output_path}, shape={data.shape}")
