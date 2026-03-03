"""GOES geostationary ↔ lat/lon reprojection utilities."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except ImportError:
    _HAS_PYPROJ = False

try:
    from pyresample import AreaDefinition, create_area_def
    from pyresample.kd_tree import resample_nearest
    from pyresample.bilinear import resample_bilinear
    _HAS_PYRESAMPLE = True
except ImportError:
    _HAS_PYRESAMPLE = False


def make_target_area(
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    resolution_deg: float,
) -> "AreaDefinition":
    """Create a pyresample AreaDefinition for the target lat/lon grid."""
    if not _HAS_PYRESAMPLE:
        raise ImportError("pyresample required for reprojection")
    n_lat = int((lat_range[1] - lat_range[0]) / resolution_deg)
    n_lon = int((lon_range[1] - lon_range[0]) / resolution_deg)
    return create_area_def(
        "goes_target",
        {"proj": "eqc", "datum": "WGS84"},
        area_extent=[lon_range[0], lat_range[0], lon_range[1], lat_range[1]],
        shape=(n_lat, n_lon),
        units="degrees",
    )


def goes_fixed_grid_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    projection_info: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GOES ABI fixed-grid coords (radians) to lat/lon.

    Args:
        x: x scan angle in radians, shape (W,)
        y: y scan angle in radians, shape (H,)
        projection_info: Dict with 'perspective_point_height',
            'semi_major_axis', 'semi_minor_axis', 'longitude_of_projection_origin'.

    Returns:
        (lats, lons) as 2D arrays of shape (H, W).
    """
    h = projection_info["perspective_point_height"]
    r_eq = projection_info.get("semi_major_axis", 6378137.0)
    r_pol = projection_info.get("semi_minor_axis", 6356752.31414)
    lon_0 = np.deg2rad(projection_info["longitude_of_projection_origin"])

    xx, yy = np.meshgrid(x, y)
    H = h + r_eq

    a = np.sin(xx) ** 2 + np.cos(xx) ** 2 * (
        np.cos(yy) ** 2 + (r_eq / r_pol) ** 2 * np.sin(yy) ** 2
    )
    b = -2.0 * H * np.cos(xx) * np.cos(yy)
    c = H**2 - r_eq**2

    discriminant = b**2 - 4 * a * c
    valid = discriminant >= 0
    discriminant = np.where(valid, discriminant, 0)

    r_s = np.where(valid, (-b - np.sqrt(discriminant)) / (2 * a), np.nan)

    sx = r_s * np.cos(xx) * np.cos(yy)
    sy = -r_s * np.sin(xx)
    sz = r_s * np.cos(xx) * np.sin(yy)

    lats = np.degrees(
        np.arctan((r_eq / r_pol) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy**2))
    )
    lons = np.degrees(lon_0 - np.arctan(sy / (H - sx)))

    lats = np.where(valid, lats, np.nan)
    lons = np.where(valid, lons, np.nan)
    return lats, lons


def reproject_to_grid(
    data: np.ndarray,
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    target_area: "AreaDefinition",
    method: str = "nearest",
    fill_value: float = np.nan,
    radius_of_influence: float = 50000,
) -> np.ndarray:
    """Reproject 2D data from source lat/lon to target AreaDefinition.

    Args:
        data: 2D array (H, W) to reproject.
        source_lats, source_lons: 2D coordinate arrays matching data shape.
        target_area: pyresample AreaDefinition for output grid.
        method: 'nearest' or 'bilinear'.
        fill_value: Value for pixels outside source domain.
        radius_of_influence: Search radius in meters.

    Returns:
        Reprojected 2D array on target grid.
    """
    if not _HAS_PYRESAMPLE:
        raise ImportError("pyresample required for reprojection")

    from pyresample.geometry import SwathDefinition
    source_def = SwathDefinition(lons=source_lons, lats=source_lats)

    if method == "bilinear":
        result = resample_bilinear(
            source_def, data, target_area,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value, nprocs=1,
        )
    else:
        result = resample_nearest(
            source_def, data, target_area,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value, nprocs=1,
        )
    return result


def compute_grid_coords(
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    resolution_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D lat and lon coordinate arrays for the target grid."""
    lats = np.arange(lat_range[0], lat_range[1], resolution_deg)
    lons = np.arange(lon_range[0], lon_range[1], resolution_deg)
    return lats, lons
