"""Visualization utilities for forecast evaluation.

Provides plotting functions for spatial forecasts, uncertainty bands,
skill curves, reliability diagrams, and spatial error maps.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import matplotlib to avoid issues when not installed
_plt = None


def _get_plt():
    global _plt
    if _plt is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            _plt = plt
        except ImportError:
            logger.warning("matplotlib not available — plotting disabled")
    return _plt


def plot_forecast_map(
    prediction: np.ndarray,
    target: np.ndarray,
    channel_names: list[str],
    timestep: int = 0,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot predicted vs observed spatial fields side-by-side."""
    plt = _get_plt()
    if plt is None:
        return

    n_ch = min(prediction.shape[0] if prediction.ndim == 3 else prediction.shape[1], 4)
    fig, axes = plt.subplots(n_ch, 2, figsize=(10, 4 * n_ch))
    if n_ch == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_ch):
        pred_i = prediction[i] if prediction.ndim == 3 else prediction[timestep, i]
        tgt_i = target[i] if target.ndim == 3 else target[timestep, i]
        name = channel_names[i] if i < len(channel_names) else f"Ch{i}"

        axes[i, 0].imshow(pred_i, cmap="RdBu_r")
        axes[i, 0].set_title(f"Predicted: {name}")
        axes[i, 1].imshow(tgt_i, cmap="RdBu_r")
        axes[i, 1].set_title(f"Observed: {name}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_uncertainty_bands(
    mean: np.ndarray,
    std: np.ndarray,
    target: np.ndarray,
    channel_idx: int = 0,
    spatial_avg: bool = True,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot forecast mean with uncertainty bands vs observation."""
    plt = _get_plt()
    if plt is None:
        return

    if spatial_avg:
        mu = mean[:, channel_idx].mean(axis=(-2, -1)) if mean.ndim == 4 else mean[:, channel_idx]
        sig = std[:, channel_idx].mean(axis=(-2, -1)) if std.ndim == 4 else std[:, channel_idx]
        obs = target[:, channel_idx].mean(axis=(-2, -1)) if target.ndim == 4 else target[:, channel_idx]
    else:
        mu, sig, obs = mean, std, target

    t = np.arange(len(mu))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, obs, "k-", label="Observed", linewidth=2)
    ax.plot(t, mu, "b-", label="Forecast mean")
    ax.fill_between(t, mu - 2 * sig, mu + 2 * sig, alpha=0.2, color="blue", label="±2σ")
    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_skill_curves(
    metrics_by_horizon: dict[int, dict[str, float]],
    metric_names: list[str] = ("rmse", "mae"),
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot metric degradation as a function of forecast horizon."""
    plt = _get_plt()
    if plt is None:
        return

    horizons = sorted(metrics_by_horizon.keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in metric_names:
        values = [metrics_by_horizon[h].get(name, float("nan")) for h in horizons]
        ax.plot(horizons, values, "o-", label=name.upper())

    ax.set_xlabel("Forecast horizon (steps)")
    ax.set_ylabel("Metric value")
    ax.set_title("Skill degradation")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    expected: np.ndarray,
    observed: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot calibration / reliability diagram."""
    plt = _get_plt()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(expected, observed, "o-", label="Model")
    ax.plot([0, 1], [0, 1], "--k", label="Perfect calibration")
    ax.set_xlabel("Expected CDF")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_spatial_error(
    error_map: np.ndarray,
    title: str = "Spatial Error",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot spatial error heatmap."""
    plt = _get_plt()
    if plt is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(error_map, cmap="hot")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)