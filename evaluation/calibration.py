"""Calibration analysis: reliability diagrams and rank histograms."""
from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats


def reliability_diagram(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute reliability diagram data.

    For each nominal probability bin, compute the actual fraction of
    observations falling below that quantile.

    Returns:
        Dict with 'nominal' and 'observed' arrays of shape (n_bins,),
        and 'ice' (integrated calibration error).
    """
    quantiles = np.linspace(0.05, 0.95, n_bins)
    observed = np.zeros(n_bins)

    sigma = np.maximum(sigma, 1e-6)

    for i, q in enumerate(quantiles):
        threshold = sp_stats.norm.ppf(q, loc=mu, scale=sigma)
        observed[i] = np.nanmean(target <= threshold)

    ice = float(np.mean(np.abs(observed - quantiles)))

    return {
        "nominal": quantiles,
        "observed": observed,
        "ice": ice,
    }


def rank_histogram(
    samples: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Compute rank histogram (Talagrand diagram).

    For each observation, find its rank among the ensemble members.
    A flat histogram indicates a well-calibrated ensemble.

    Args:
        samples: (N, ...) ensemble of N samples.
        target: (...) observations.

    Returns:
        Histogram counts of shape (N+1,).
    """
    N = samples.shape[0]
    flat_samples = samples.reshape(N, -1)  # (N, M)
    flat_target = target.reshape(1, -1)    # (1, M)

    # Count how many ensemble members are below the observation
    ranks = np.sum(flat_samples < flat_target, axis=0)  # (M,)
    hist, _ = np.histogram(ranks, bins=np.arange(N + 2) - 0.5)
    return hist


def sharpness(sigma: np.ndarray, confidence: float = 0.8) -> float:
    """Mean width of the CI across all forecasts.

    Args:
        sigma: Predicted standard deviations.
        confidence: CI level (default 80%).

    Returns:
        Mean CI width.
    """
    z = sp_stats.norm.ppf(0.5 + confidence / 2)
    widths = 2 * z * sigma
    return float(np.nanmean(widths))


def calibration_summary(
    mu: np.ndarray,
    sigma: np.ndarray,
    target: np.ndarray,
    confidence_levels: list[float] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, dict[str, float]]:
    """Compute calibration summary: nominal vs actual coverage at each level."""
    from evaluation.metrics import coverage_score

    results = {}
    for cl in confidence_levels:
        actual = coverage_score(mu, sigma, target, confidence=cl)
        results[f"{int(cl*100)}%"] = {
            "nominal": cl,
            "actual": actual,
            "gap": abs(actual - cl),
        }
    return results
