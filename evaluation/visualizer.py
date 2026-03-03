"""Visualization utilities for forecast evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _ensure_mpl():
    if not _HAS_MPL:
        raise ImportError("matplotlib required for visualization")


def plot_forecast_map(
    pred: np.ndarray,
    target: np.ndarray,
    channel_name: str = "",
    save_path: Optional[str | Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Side-by-side forecast vs truth with error overlay.

    Args:
        pred, target: 2D arrays (H, W).
        channel_name: Label for the channel.
        save_path: If provided, saves figure to this path.
    """
    _ensure_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(pred, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Predicted {channel_name}")
    plt.colorbar(im0, ax=axes[0], shrink=0.7)

    im1 = axes[1].imshow(target, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Observed {channel_name}")
    plt.colorbar(im1, ax=axes[1], shrink=0.7)

    error = pred - target
    elim = max(abs(np.nanmin(error)), abs(np.nanmax(error))) or 1.0
    im2 = axes[2].imshow(error, cmap="RdBu_r", vmin=-elim, vmax=elim)
    axes[2].set_title("Error (Pred - Obs)")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_uncertainty_bands(
    mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    observed: Optional[np.ndarray] = None,
    channel_name: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """Time series with shaded confidence interval.

    Args:
        mean, lower, upper: 1D arrays (T,) — forecast mean and CI bounds.
        observed: 1D array (T,) — ground truth.
    """
    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.arange(len(mean))

    ax.fill_between(t, lower, upper, alpha=0.3, color="steelblue", label="90% CI")
    ax.plot(t, mean, color="steelblue", linewidth=2, label="Forecast mean")
    if observed is not None:
        ax.plot(t, observed, "k--", linewidth=1.5, label="Observed")

    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(channel_name or "Value")
    ax.set_title(f"Forecast with Uncertainty — {channel_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_skill_curves(
    results: dict[str, dict[int, float]],
    metric_name: str = "RMSE",
    save_path: Optional[str | Path] = None,
) -> None:
    """Skill metric vs lead time for multiple variables.

    Args:
        results: {variable_name: {lead_time: metric_value}}
        metric_name: Name of the metric being plotted.
    """
    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(8, 5))

    for var_name, scores in results.items():
        lead_times = sorted(scores.keys())
        values = [scores[lt] for lt in lead_times]
        ax.plot(lead_times, values, "o-", label=var_name, linewidth=2, markersize=5)

    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs Lead Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_reliability_diagram(
    nominal: np.ndarray,
    observed: np.ndarray,
    ice: float = 0.0,
    save_path: Optional[str | Path] = None,
) -> None:
    """Reliability diagram: nominal vs observed quantile coverage.

    Args:
        nominal: Array of nominal probability levels.
        observed: Array of observed hit rates at each level.
        ice: Integrated calibration error.
    """
    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(nominal, observed, "o-", color="steelblue", linewidth=2, label=f"Model (ICE={ice:.3f})")

    ax.set_xlabel("Nominal probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_spatial_error(
    error_map: np.ndarray,
    channel_name: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """Geographic heatmap of mean absolute error."""
    _ensure_mpl()
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(error_map, cmap="YlOrRd")
    ax.set_title(f"Spatial MAE — {channel_name}")
    plt.colorbar(im, ax=ax, shrink=0.7, label="MAE")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
