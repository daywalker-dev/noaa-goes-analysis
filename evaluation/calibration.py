"""Calibration analysis for probabilistic forecasts.

Provides reliability diagrams, rank histograms, and sharpness metrics
to assess whether predicted uncertainty is well-calibrated.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def reliability_diagram(
    predicted_probs: np.ndarray,
    observed_freqs: np.ndarray,
    n_bins: int = 20,
) -> dict[str, np.ndarray]:
    """Compute reliability diagram data (expected vs observed CDF).

    Args:
        predicted_probs: (N,) PIT values or predicted CDF values.
        observed_freqs: (N,) binary indicators (1 if observed <= predicted).
        n_bins: Number of bins for the diagram.

    Returns:
        Dict with 'expected', 'observed', 'counts' arrays.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    expected = []
    observed = []
    counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (predicted_probs >= lo) & (predicted_probs < hi)
        n = mask.sum()
        if n > 0:
            expected.append((lo + hi) / 2)
            observed.append(observed_freqs[mask].mean())
            counts.append(int(n))

    return {
        "expected": np.array(expected),
        "observed": np.array(observed),
        "counts": np.array(counts),
    }


def rank_histogram(
    ensemble_samples: np.ndarray,
    observations: np.ndarray,
) -> np.ndarray:
    """Compute rank histogram for ensemble forecasts.

    Args:
        ensemble_samples: (N, K) ensemble members.
        observations: (N,) observed values.

    Returns:
        Histogram counts of shape (K+1,).
    """
    N, K = ensemble_samples.shape
    ranks = np.zeros(N, dtype=int)

    for i in range(N):
        # Where does the observation rank among the ensemble?
        ranks[i] = np.searchsorted(np.sort(ensemble_samples[i]), observations[i])

    hist, _ = np.histogram(ranks, bins=np.arange(K + 2))
    return hist


def sharpness(
    predicted_std: np.ndarray,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, float]:
    """Compute sharpness metrics (narrower intervals = sharper, if calibrated).

    Args:
        predicted_std: (N,) predicted standard deviations.
        quantiles: Quantiles to report.

    Returns:
        Dict with 'mean_std', 'median_std', and requested quantiles.
    """
    valid = predicted_std[np.isfinite(predicted_std)]
    result = {
        "mean_std": float(np.mean(valid)),
        "median_std": float(np.median(valid)),
    }
    for q in quantiles:
        result[f"q{int(q*100)}_std"] = float(np.quantile(valid, q))
    return result


def calibration_summary(
    predicted_mean: np.ndarray,
    predicted_std: np.ndarray,
    observations: np.ndarray,
    confidence_levels: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
    n_bins: int = 20,
) -> dict:
    """Full calibration summary: PIT histogram, coverage, sharpness.

    Args:
        predicted_mean: (N,) or (N,...) predicted means.
        predicted_std: (N,) or (N,...) predicted stds.
        observations: (N,) or (N,...) observed values.
        confidence_levels: Confidence levels to compute coverage for.
        n_bins: Bins for reliability diagram.

    Returns:
        Dict with 'pit_histogram', 'coverage', 'reliability', 'sharpness'.
    """
    # Flatten everything
    mu = predicted_mean.ravel()
    sigma = predicted_std.ravel()
    obs = observations.ravel()

    # Remove invalid entries
    valid = np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(obs) & (sigma > 1e-8)
    mu, sigma, obs = mu[valid], sigma[valid], obs[valid]

    if len(mu) == 0:
        logger.warning("No valid data for calibration summary")
        return {}

    # PIT values: Φ((obs - μ) / σ)
    from scipy.stats import norm
    pit_values = norm.cdf((obs - mu) / sigma)

    # PIT histogram (should be uniform if well-calibrated)
    pit_hist, _ = np.histogram(pit_values, bins=n_bins, range=(0, 1))

    # Coverage for each confidence level
    coverage = {}
    for level in confidence_levels:
        z = norm.ppf(0.5 + level / 2)
        in_interval = np.abs(obs - mu) <= z * sigma
        coverage[f"{int(level*100)}%"] = float(in_interval.mean())

    # Reliability diagram
    pit_binary = (pit_values <= np.linspace(0.05, 0.95, n_bins)[:, None]).mean(axis=1)
    rel = reliability_diagram(
        np.repeat(np.linspace(0.05, 0.95, n_bins), len(pit_values) // n_bins + 1)[:len(pit_values)],
        (pit_values <= 0.5).astype(float),
        n_bins=n_bins,
    )

    return {
        "pit_histogram": pit_hist.tolist(),
        "coverage": coverage,
        "reliability": {k: v.tolist() for k, v in rel.items()},
        "sharpness": sharpness(sigma),
    }