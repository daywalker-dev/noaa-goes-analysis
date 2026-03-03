"""Evaluation metrics for deterministic, probabilistic, and spatial forecast quality."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats as sp_stats


def rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Root mean squared error."""
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff[mask > 0]
    return float(np.sqrt(np.nanmean(diff)))


def mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean absolute error."""
    diff = np.abs(pred - target)
    if mask is not None:
        diff = diff[mask > 0]
    return float(np.nanmean(diff))


def bias(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """Mean bias (pred - target)."""
    diff = pred - target
    if mask is not None:
        diff = diff[mask > 0]
    return float(np.nanmean(diff))


def ssim_score(
    pred: np.ndarray,
    target: np.ndarray,
    data_range: float = 1.0,
    win_size: int = 7,
) -> float:
    """Structural similarity index (simplified implementation).

    Args:
        pred, target: 2D arrays (H, W) or 3D (C, H, W).
        data_range: Dynamic range of the data.
        win_size: Window size for local statistics.
    """
    from scipy.ndimage import uniform_filter

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    mu_x = uniform_filter(pred, size=win_size)
    mu_y = uniform_filter(target, size=win_size)

    sigma_x_sq = uniform_filter(pred ** 2, size=win_size) - mu_x ** 2
    sigma_y_sq = uniform_filter(target ** 2, size=win_size) - mu_y ** 2
    sigma_xy = uniform_filter(pred * target, size=win_size) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
    )
    return float(np.nanmean(ssim_map))


def spatial_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Anomaly correlation coefficient (ACC).

    Measures spatial pattern similarity after removing the mean.
    """
    pred_anom = pred - np.nanmean(pred)
    target_anom = target - np.nanmean(target)
    num = np.nansum(pred_anom * target_anom)
    denom = np.sqrt(np.nansum(pred_anom ** 2) * np.nansum(target_anom ** 2))
    if denom < 1e-10:
        return 0.0
    return float(num / denom)


def crps_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: np.ndarray,
) -> float:
    """Closed-form CRPS for Gaussian distribution N(mu, sigma^2).

    CRPS = σ [y_norm (2Φ(y_norm) - 1) + 2φ(y_norm) - 1/√π]
    """
    sigma = np.maximum(sigma, 1e-6)
    y_norm = (target - mu) / sigma
    phi = sp_stats.norm.pdf(y_norm)
    Phi = sp_stats.norm.cdf(y_norm)
    crps_vals = sigma * (y_norm * (2 * Phi - 1) + 2 * phi - 1.0 / math.sqrt(math.pi))
    return float(np.nanmean(crps_vals))


def crps_ensemble(samples: np.ndarray, target: np.ndarray) -> float:
    """CRPS estimated from ensemble samples using energy form.

    Args:
        samples: (N, ...) ensemble of N samples.
        target: (...) observations.
    """
    N = samples.shape[0]
    # E|X - y|
    term1 = np.nanmean(np.abs(samples - target[None]), axis=0)
    # E|X - X'| / 2
    term2 = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            term2 = term2 + np.nanmean(np.abs(samples[i] - samples[j]))
    term2 = term2 / (N * (N - 1) / 2) / 2 if N > 1 else 0.0
    return float(np.nanmean(term1) - term2)


def coverage_score(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: np.ndarray,
    confidence: float = 0.9,
) -> float:
    """Fraction of observations falling within the CI.

    Args:
        mu, sigma: Predicted mean and std.
        target: Observed values.
        confidence: Confidence level (e.g. 0.9 for 90% CI).
    """
    z = sp_stats.norm.ppf(0.5 + confidence / 2)
    lower = mu - z * sigma
    upper = mu + z * sigma
    in_interval = (target >= lower) & (target <= upper)
    return float(np.nanmean(in_interval))


def multistep_skill(
    pred_sequence: np.ndarray,
    target_sequence: np.ndarray,
    metric_fn,
    lead_times: Optional[list[int]] = None,
) -> dict[int, float]:
    """Compute a metric at each lead time.

    Args:
        pred_sequence: (T, ...) predictions at each forecast step.
        target_sequence: (T, ...) observations at each step.
        metric_fn: Callable(pred, target) → float.
        lead_times: Specific lead times to evaluate (1-indexed). None = all.

    Returns:
        {lead_time: metric_value}
    """
    T = pred_sequence.shape[0]
    if lead_times is None:
        lead_times = list(range(1, T + 1))

    results = {}
    for lt in lead_times:
        if lt <= T:
            results[lt] = metric_fn(pred_sequence[lt - 1], target_sequence[lt - 1])
    return results


# ---------------------------------------------------------------------------
# Metric registry for config-driven evaluation
# ---------------------------------------------------------------------------
METRIC_REGISTRY = {
    "rmse": rmse,
    "mae": mae,
    "bias": bias,
    "ssim": ssim_score,
    "spatial_correlation": spatial_correlation,
    "crps_gaussian": crps_gaussian,
    "coverage": coverage_score,
}


def get_metric(name: str):
    """Retrieve metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name]
