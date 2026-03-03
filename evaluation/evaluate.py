#!/usr/bin/env python
"""Run full evaluation suite on a trained model checkpoint."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import numpy as np
import torch
from omegaconf import OmegaConf

from goes_forecast.utils.config_loader import load_config
from goes_forecast.utils.logger import get_logger
from goes_forecast.utils.reproducibility import set_global_seed
from goes_forecast.data.dataset import build_dataloader
from goes_forecast.models.spatial_encoder import DomainEncoderEnsemble
from goes_forecast.models.temporal_bayesian import VariationalTransformer
from goes_forecast.models.reverse_generator import ConditionalUNet
from goes_forecast.models.fusion import FusionTransformer
from goes_forecast.evaluation.metrics import (
    rmse, mae, bias, ssim_score, spatial_correlation,
    crps_gaussian, coverage_score, multistep_skill, get_metric,
)
from goes_forecast.evaluation.calibration import (
    reliability_diagram, calibration_summary, rank_histogram, sharpness,
)
from goes_forecast.evaluation.visualizer import (
    plot_forecast_map, plot_uncertainty_bands,
    plot_skill_curves, plot_reliability_diagram, plot_spatial_error,
)

logger = get_logger(__name__)


class PersistenceBaseline:
    """Trivial baseline: repeat last observed frame for all forecast steps."""

    def predict(self, input_data: np.ndarray, n_steps: int) -> np.ndarray:
        """input_data: (T_in, C, H, W). Returns (n_steps, C, H, W)."""
        last_frame = input_data[-1:]  # (1, C, H, W)
        return np.repeat(last_frame, n_steps, axis=0)


def _get_domain_channels(cfg) -> dict[str, int]:
    domain_ch: dict[str, int] = {}
    for product in cfg.data.products:
        d = product["domain"]
        if d == "meteo":
            continue
        domain_ch[d] = domain_ch.get(d, 0) + len(product["channels"])
    return domain_ch


def _get_domain_indices(cfg) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    offset = 0
    for product in cfg.data.products:
        d = product["domain"]
        n = len(product["channels"])
        indices.setdefault(d, []).extend(range(offset, offset + n))
        offset += n
    return indices


def _total_channels(cfg) -> int:
    return sum(len(p["channels"]) for p in cfg.data.products)


def _all_channel_names(cfg) -> list[str]:
    names = []
    for product in cfg.data.products:
        names.extend(product["channels"])
    return names


def load_models(cfg, checkpoint_dir: Path, device: torch.device):
    """Load all four model stages from checkpoints."""
    domain_ch = _get_domain_channels(cfg)
    domain_idx = _get_domain_indices(cfg)
    meteo_dim = len(domain_idx.get("meteo", []))
    total_ch = _total_channels(cfg)

    encoders = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
    temporal = VariationalTransformer(
        latent_dim=encoders.combined_latent_dim,
        meteo_dim=max(meteo_dim, 1),
        state_dim=cfg.model.temporal.state_dim,
        d_model=cfg.model.temporal.d_model,
        n_heads=cfg.model.temporal.n_heads,
        n_encoder_layers=cfg.model.temporal.n_encoder_layers,
        n_decoder_layers=cfg.model.temporal.n_decoder_layers,
        dim_feedforward=cfg.model.temporal.dim_feedforward,
        dropout=cfg.model.temporal.dropout,
        forecast_steps=cfg.data.temporal.forecast_steps,
        beta_kl=cfg.model.temporal.beta_kl,
        free_bits=cfg.model.temporal.free_bits,
    )
    generator = ConditionalUNet(
        state_dim=cfg.model.temporal.state_dim,
        out_channels=total_ch,
        noise_dim=cfg.model.generator.noise_dim,
        base_channels=cfg.model.generator.base_channels,
        channel_multipliers=list(cfg.model.generator.channel_multipliers),
        n_res_blocks=cfg.model.generator.n_res_blocks,
        use_spectral_norm=cfg.model.generator.use_spectral_norm,
        initial_spatial=tuple(cfg.model.generator.initial_spatial),
    )
    fusion = FusionTransformer(
        cnn_dim=encoders.combined_latent_dim,
        bayes_dim=cfg.model.temporal.state_dim * 2,
        meteo_dim=max(meteo_dim, 1),
        gen_dim=total_ch,
        d_model=cfg.model.fusion.d_model,
        n_heads=cfg.model.fusion.n_heads,
        n_layers=cfg.model.fusion.n_layers,
        dim_feedforward=cfg.model.fusion.dim_feedforward,
        dropout=cfg.model.fusion.dropout,
        out_channels=total_ch,
        spatial_size=(cfg.data.spatial.patch_size, cfg.data.spatial.patch_size),
    )

    # Load checkpoints
    for name, model, subdir in [
        ("encoders", encoders, "encoders"),
        ("temporal", temporal, "temporal"),
        ("generator", generator, "generator"),
        ("fusion", fusion, "fusion"),
    ]:
        ckpt_path = checkpoint_dir / subdir / "checkpoints" / "best.ckpt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Handle wrapped ModuleDict for fusion
            sd = state.get("state_dict", state)
            try:
                model.load_state_dict(sd, strict=False)
                logger.info(f"Loaded {name} from {ckpt_path}")
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}")

    for m in [encoders, temporal, generator, fusion]:
        m.to(device).eval()

    return encoders, temporal, generator, fusion


@torch.no_grad()
def run_inference(
    encoders, temporal, generator, fusion,
    batch: dict, cfg, domain_idx: dict, device: torch.device,
    n_mc_samples: int = 50,
) -> dict[str, np.ndarray]:
    """Run full pipeline inference on a batch, returning numpy arrays."""
    x = batch["input"].to(device)
    meteo_in = batch["meteo_input"].to(device)
    meteo_tgt = batch["meteo_target"].to(device)
    B, T_in, C, H, W = x.shape
    T_out = cfg.data.temporal.forecast_steps

    # Extract latents
    latents = []
    for t in range(T_in):
        frame = x[:, t]
        d_inputs = {
            d: frame[:, idx] for d, idx in domain_idx.items()
            if idx and d != "meteo"
        }
        latents.append(encoders.encode_only(d_inputs))
    latents_stack = torch.stack(latents, dim=1)

    # MC sampling for uncertainty
    mc_outputs = temporal.sample(latents_stack, meteo_in, n_samples=n_mc_samples)
    mu = mc_outputs["mean"]  # (B, T_out, state_dim)
    std = mc_outputs["std"]
    p05 = mc_outputs["p05"]
    p95 = mc_outputs["p95"]

    # Generator
    gen_spatial = generator.decode_sequence(mu, target_size=(H, W))

    # Fusion
    last_latent = latents_stack[:, -1:].expand(-1, T_out, -1)
    temporal_out = temporal(latents_stack, meteo_in)
    bayes_feat = torch.cat([temporal_out["mu"], temporal_out["logvar"]], dim=-1)
    gen_feat = gen_spatial.mean(dim=(-2, -1))

    fusion_out = fusion(
        cnn_latents=last_latent,
        bayes_output=bayes_feat,
        meteo_fields=meteo_tgt,
        gen_features=gen_feat,
        gen_spatial=gen_spatial,
    )

    # Use fusion forecast if available, otherwise generator output
    forecast = fusion_out.get("forecast", gen_spatial)
    uncertainty = fusion_out.get("uncertainty", None)

    result = {
        "forecast": forecast.cpu().numpy(),
        "target": batch["target"].numpy(),
        "mu": mu.cpu().numpy(),
        "std": std.cpu().numpy(),
        "p05": p05.cpu().numpy(),
        "p95": p95.cpu().numpy(),
        "gen_spatial": gen_spatial.cpu().numpy(),
    }
    if uncertainty is not None:
        result["uncertainty"] = uncertainty.cpu().numpy()
    return result


@click.command()
@click.option("--checkpoint", required=True, help="Path to experiment output dir")
@click.option("--config", "config_path", default=None, help="Config path (defaults to checkpoint dir)")
@click.option("--split", default="test", help="Data split to evaluate")
@click.option("--lead-times", default="1,3,6,12,24", help="Comma-separated lead times")
@click.option("--n-samples", default=50, type=int, help="MC samples for uncertainty")
@click.option("--output-dir", default=None, help="Output directory for results")
@click.option("--max-batches", default=None, type=int, help="Max batches to evaluate (for speed)")
def main(checkpoint, config_path, split, lead_times, n_samples, output_dir, max_batches):
    checkpoint_dir = Path(checkpoint)
    if config_path is None:
        config_path = checkpoint_dir / "config.yaml"
    cfg = load_config(config_path)
    set_global_seed(cfg.project.seed)

    output_path = Path(output_dir) if output_dir else checkpoint_dir / "evaluation"
    output_path.mkdir(parents=True, exist_ok=True)

    lead_time_list = [int(x) for x in lead_times.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain_idx = _get_domain_indices(cfg)
    channel_names = _all_channel_names(cfg)

    logger.info(f"Evaluating on {split} split, device={device}")
    logger.info(f"Lead times: {lead_time_list}, MC samples: {n_samples}")

    # Load models
    encoders, temporal, generator, fusion = load_models(cfg, checkpoint_dir, device)

    # Build dataloader
    test_loader = build_dataloader(cfg, split=split)

    # Collect predictions
    all_forecasts, all_targets = [], []
    all_mu, all_std = [], []
    baseline = PersistenceBaseline()
    all_persistence = []

    for batch_idx, batch in enumerate(test_loader):
        if max_batches and batch_idx >= max_batches:
            break

        result = run_inference(
            encoders, temporal, generator, fusion,
            batch, cfg, domain_idx, device, n_samples,
        )

        all_forecasts.append(result["forecast"])
        all_targets.append(result["target"])
        all_mu.append(result["mu"])
        all_std.append(result["std"])

        # Persistence baseline
        input_np = batch["input"].numpy()
        for b in range(input_np.shape[0]):
            all_persistence.append(
                baseline.predict(input_np[b], cfg.data.temporal.forecast_steps)
            )

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1} batches")

    if not all_forecasts:
        logger.error("No data processed. Check data path and split.")
        return

    # Stack all results: (N, T_out, C, H, W) for forecasts/targets
    forecasts = np.concatenate(all_forecasts, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    mu_states = np.concatenate(all_mu, axis=0)
    std_states = np.concatenate(all_std, axis=0)
    persistence = np.stack(all_persistence, axis=0)

    N, T_out, C, H, W = forecasts.shape
    logger.info(f"Evaluation data: {N} samples, {T_out} forecast steps, {C} channels")

    # ---- Compute deterministic metrics per lead time per channel ----
    results_by_metric: dict[str, dict[str, dict[int, float]]] = {}
    persistence_by_metric: dict[str, dict[str, dict[int, float]]] = {}

    for metric_name in cfg.evaluation.metrics:
        if metric_name in ("crps_gaussian", "coverage"):
            continue  # Handled separately
        metric_fn = get_metric(metric_name)
        results_by_metric[metric_name] = {}
        persistence_by_metric[metric_name] = {}

        for c_idx, c_name in enumerate(channel_names):
            if c_idx >= C:
                break
            pred_ch = forecasts[:, :, c_idx]   # (N, T, H, W)
            tgt_ch = targets[:, :, c_idx]
            pers_ch = persistence[:, :, c_idx]

            scores = {}
            pers_scores = {}
            for lt in lead_time_list:
                if lt <= T_out:
                    t_idx = lt - 1
                    scores[lt] = metric_fn(pred_ch[:, t_idx], tgt_ch[:, t_idx])
                    pers_scores[lt] = metric_fn(pers_ch[:, t_idx], tgt_ch[:, t_idx])
            results_by_metric[metric_name][c_name] = scores
            persistence_by_metric[metric_name][c_name] = pers_scores

    # ---- Probabilistic metrics (CRPS, coverage) on state-level ----
    state_target = targets.mean(axis=(-2, -1))  # (N, T, C)
    crps_results = {}
    coverage_results = {}

    for c_idx, c_name in enumerate(channel_names):
        if c_idx >= mu_states.shape[-1]:
            break
        mu_c = mu_states[:, :, c_idx]   # (N, T)
        std_c = std_states[:, :, c_idx]
        tgt_c = state_target[:, :, min(c_idx, state_target.shape[-1] - 1)]

        crps_scores = {}
        cov_scores = {}
        for lt in lead_time_list:
            if lt <= T_out:
                t = lt - 1
                crps_scores[lt] = crps_gaussian(mu_c[:, t], std_c[:, t], tgt_c[:, t])
                cov_scores[lt] = coverage_score(mu_c[:, t], std_c[:, t], tgt_c[:, t], confidence=0.9)
        crps_results[c_name] = crps_scores
        coverage_results[c_name] = cov_scores

    # ---- Calibration analysis ----
    flat_mu = mu_states.reshape(-1)
    flat_std = std_states.reshape(-1)
    flat_tgt = np.broadcast_to(
        state_target[..., :mu_states.shape[-1]], mu_states.shape
    ).reshape(-1)

    reliability = reliability_diagram(flat_mu, flat_std, flat_tgt)
    cal_summary = calibration_summary(flat_mu, flat_std, flat_tgt)
    sharp = sharpness(flat_std)

    # ---- Save results ----
    all_results = {
        "deterministic": {
            metric: {ch: {str(k): v for k, v in scores.items()}
                     for ch, scores in channels.items()}
            for metric, channels in results_by_metric.items()
        },
        "persistence_baseline": {
            metric: {ch: {str(k): v for k, v in scores.items()}
                     for ch, scores in channels.items()}
            for metric, channels in persistence_by_metric.items()
        },
        "crps": {ch: {str(k): v for k, v in s.items()} for ch, s in crps_results.items()},
        "coverage_90": {ch: {str(k): v for k, v in s.items()} for ch, s in coverage_results.items()},
        "calibration": cal_summary,
        "sharpness_80": sharp,
        "reliability_ice": reliability["ice"],
        "n_samples": N,
        "lead_times": lead_time_list,
    }

    results_path = output_path / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # ---- Generate plots ----
    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        # Skill curves
        for metric_name, channels in results_by_metric.items():
            plot_skill_curves(
                channels,
                metric_name=metric_name.upper(),
                save_path=plots_dir / f"skill_{metric_name}.png",
            )

        # CRPS skill curves
        if crps_results:
            plot_skill_curves(crps_results, metric_name="CRPS", save_path=plots_dir / "skill_crps.png")

        # Reliability diagram
        plot_reliability_diagram(
            reliability["nominal"], reliability["observed"],
            ice=reliability["ice"],
            save_path=plots_dir / "reliability.png",
        )

        # Spatial error map (first channel, last lead time)
        if forecasts.shape[0] > 0:
            error_map = np.mean(np.abs(forecasts[:, -1, 0] - targets[:, -1, 0]), axis=0)
            plot_spatial_error(
                error_map,
                channel_name=channel_names[0] if channel_names else "",
                save_path=plots_dir / "spatial_error.png",
            )

        # Forecast map example (first sample)
        if forecasts.shape[0] > 0:
            plot_forecast_map(
                forecasts[0, -1, 0], targets[0, -1, 0],
                channel_name=channel_names[0] if channel_names else "",
                save_path=plots_dir / "forecast_example.png",
            )

        # Uncertainty bands (spatially averaged first channel)
        if mu_states.shape[0] > 0 and mu_states.shape[-1] > 0:
            idx = 0
            mean_ts = mu_states[0, :, idx]
            p05_ts = mu_states[0, :, idx] - 1.645 * std_states[0, :, idx]
            p95_ts = mu_states[0, :, idx] + 1.645 * std_states[0, :, idx]
            obs_ts = flat_tgt[:T_out] if len(flat_tgt) >= T_out else None
            plot_uncertainty_bands(
                mean_ts, p05_ts, p95_ts,
                channel_name=channel_names[0] if channel_names else "",
                save_path=plots_dir / "uncertainty_bands.png",
            )

        logger.info(f"Plots saved to {plots_dir}")
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    # ---- Print summary ----
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for metric_name, channels in results_by_metric.items():
        for ch_name, scores in channels.items():
            score_str = " | ".join(f"{lt}h: {v:.4f}" for lt, v in sorted(scores.items()))
            logger.info(f"  {metric_name:>20s} / {ch_name}: {score_str}")

    logger.info(f"\n  Calibration (90% CI coverage): {cal_summary}")
    logger.info(f"  Reliability ICE: {reliability['ice']:.4f}")
    logger.info(f"  Sharpness (80% CI width): {sharp:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
