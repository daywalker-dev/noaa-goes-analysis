"""
Evaluation Suite
=================
Comprehensive metrics for deterministic and probabilistic forecast evaluation.

Metrics:
    - RMSE / MAE
    - SSIM
    - CRPS (Continuous Ranked Probability Score)
    - Calibration curves
    - Spatial correlation
    - Multi-step degradation analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# Deterministic Metrics
# ===========================================================================

def rmse(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Root Mean Squared Error over valid pixels."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean Absolute Error over valid pixels."""
    if mask is not None:
        pred, target = pred[mask], target[mask]
    return float(np.mean(np.abs(pred - target)))


def ssim_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> float:
    """Compute mean SSIM (higher is better). Input: (B, C, H, W) tensors."""
    from models.blocks import ssim_loss
    # ssim_loss returns 1 - SSIM
    return float(1.0 - ssim_loss(pred, target, window_size))


def spatial_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation computed over spatial dims.

    Input: (H, W) or (C, H, W) arrays.  Returns mean correlation across channels.
    """
    if pred.ndim == 2:
        pred = pred[np.newaxis]
        target = target[np.newaxis]

    corrs = []
    for c in range(pred.shape[0]):
        p = pred[c].ravel()
        t = target[c].ravel()
        valid = np.isfinite(p) & np.isfinite(t)
        if valid.sum() < 2:
            continue
        p, t = p[valid], t[valid]
        r = np.corrcoef(p, t)[0, 1]
        if np.isfinite(r):
            corrs.append(r)
    return float(np.mean(corrs)) if corrs else 0.0


# ===========================================================================
# Probabilistic Metrics
# ===========================================================================

def crps_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    obs: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score for Gaussian forecasts.

    Closed-form for N(mu, sigma²):
        CRPS = sigma * [ z*Phi(z) + phi(z) - 1/sqrt(pi) ]
    where z = (obs - mu) / sigma.
    """
    from scipy.stats import norm

    sigma = np.clip(sigma, 1e-6, None)
    z = (obs - mu) / sigma
    crps_values = sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )
    return float(np.mean(crps_values))


def crps_ensemble(
    samples: np.ndarray,
    obs: np.ndarray,
) -> float:
    """CRPS from ensemble samples using the PWM estimator.

    Parameters
    ----------
    samples : (num_samples, ...) — ensemble forecasts
    obs     : (...) — observations (same trailing shape)
    """
    S = samples.shape[0]
    # Term 1: E|X - y|
    term1 = np.mean(np.abs(samples - obs[np.newaxis]), axis=0)
    # Term 2: E|X - X'| / 2
    term2 = np.zeros_like(term1)
    for i in range(S):
        for j in range(i + 1, S):
            term2 += np.abs(samples[i] - samples[j])
    term2 /= (S * (S - 1) / 2) if S > 1 else 1.0
    return float(np.mean(term1 - 0.5 * term2))


# ===========================================================================
# Calibration
# ===========================================================================

def calibration_curve(
    mu: np.ndarray,
    sigma: np.ndarray,
    obs: np.ndarray,
    num_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reliability diagram data for probabilistic calibration.

    Returns (expected_freq, observed_freq) arrays of shape (num_bins,).
    A well-calibrated model has observed_freq ≈ expected_freq.
    """
    from scipy.stats import norm

    sigma = np.clip(sigma, 1e-6, None)
    cdf_values = norm.cdf(obs, loc=mu, scale=sigma).ravel()

    bin_edges = np.linspace(0, 1, num_bins + 1)
    expected = np.zeros(num_bins)
    observed = np.zeros(num_bins)

    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (cdf_values >= lo) & (cdf_values < hi)
        expected[i] = (lo + hi) / 2.0
        observed[i] = mask.mean() if mask.any() else 0.0

    return expected, observed


# ===========================================================================
# Multi-step Degradation Analysis
# ===========================================================================

def multistep_degradation(
    model_forecast_fn,
    init_state: torch.Tensor,
    targets: List[torch.Tensor],
    horizons: List[int],
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    """Evaluate forecast quality at each horizon step.

    Parameters
    ----------
    model_forecast_fn : callable
        ``(state, steps) -> dict(mean, std)``
    init_state : (B, D)
    targets : list of (B, D) tensors, one per future step
    horizons : list of int step indices to evaluate

    Returns
    -------
    dict mapping horizon → {rmse, mae, spatial_corr}
    """
    results: Dict[int, Dict[str, float]] = {}
    max_h = max(horizons)
    forecast = model_forecast_fn(init_state.to(device), max_h)
    pred_mean = forecast["mean"].cpu().numpy()  # (B, steps, D)

    for h in horizons:
        if h > len(targets):
            continue
        t_np = targets[h - 1].cpu().numpy()
        p_np = pred_mean[:, h - 1]
        results[h] = {
            "rmse": rmse(p_np, t_np),
            "mae": mae(p_np, t_np),
        }
    return results


# ===========================================================================
# Full Evaluation Pipeline
# ===========================================================================

def run_evaluation(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all configured metrics and return a results dictionary.

    Parameters
    ----------
    predictions : dict with keys "mean", "std", "samples" (optional)
    targets : dict with key "obs"
    cfg : full project config

    Returns
    -------
    dict of metric name → value
    """
    eval_cfg = cfg.get("evaluation", {})
    metrics_list = eval_cfg.get("metrics", ["rmse", "mae"])
    results: Dict[str, Any] = {}

    pred_mu = predictions["mean"]
    obs = targets["obs"]

    if "rmse" in metrics_list:
        results["rmse"] = rmse(pred_mu, obs)
    if "mae" in metrics_list:
        results["mae"] = mae(pred_mu, obs)
    if "ssim" in metrics_list:
        # Convert to torch for SSIM
        p_t = torch.from_numpy(pred_mu).float()
        o_t = torch.from_numpy(obs).float()
        if p_t.ndim == 3:
            p_t = p_t.unsqueeze(0)
            o_t = o_t.unsqueeze(0)
        results["ssim"] = ssim_metric(p_t, o_t)
    if "spatial_correlation" in metrics_list:
        results["spatial_correlation"] = spatial_correlation(pred_mu, obs)
    if "crps" in metrics_list and "std" in predictions:
        results["crps"] = crps_gaussian(pred_mu, predictions["std"], obs)

    # Calibration
    if "std" in predictions:
        num_bins = eval_cfg.get("calibration", {}).get("num_bins", 20)
        expected, observed = calibration_curve(
            pred_mu, predictions["std"], obs, num_bins
        )
        results["calibration"] = {"expected": expected.tolist(), "observed": observed.tolist()}

    logger.info("Evaluation results: %s",
                {k: v for k, v in results.items() if k != "calibration"})
    return results


# ===========================================================================
# Optional: save evaluation plots
# ===========================================================================

def save_evaluation_plots(
    results: Dict[str, Any],
    plot_dir: str | Path,
) -> None:
    """Generate and save evaluation plots (requires matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot generation")
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Calibration plot
    cal = results.get("calibration")
    if cal:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(cal["expected"], cal["observed"], "o-", label="Model")
        ax.plot([0, 1], [0, 1], "--k", label="Perfect")
        ax.set_xlabel("Expected CDF")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "calibration.png", dpi=150)
        plt.close(fig)
        logger.info("Saved calibration plot → %s", plot_dir / "calibration.png")

    # Bar chart of scalar metrics
    scalar = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    if scalar:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(scalar.keys(), scalar.values())
        ax.set_ylabel("Metric value")
        ax.set_title("Evaluation Metrics")
        fig.tight_layout()
        fig.savefig(plot_dir / "metrics_summary.png", dpi=150)
        plt.close(fig)