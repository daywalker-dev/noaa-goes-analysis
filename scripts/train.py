#!/usr/bin/env python
"""Orchestrate multi-stage training of the GOES forecast pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import click
import torch
from omegaconf import OmegaConf

from goes_forecast.utils.config_loader import (
    load_config, validate_config, generate_experiment_id, save_config,
)
from goes_forecast.utils.logger import get_logger
from goes_forecast.utils.reproducibility import set_global_seed, get_environment_info
from goes_forecast.data.dataset import build_dataloader
from goes_forecast.models.spatial_encoder import DomainEncoderEnsemble
from goes_forecast.models.temporal_bayesian import VariationalTransformer
from goes_forecast.models.reverse_generator import ConditionalUNet
from goes_forecast.models.fusion import FusionTransformer
from goes_forecast.training.stage_runners import (
    EncoderStageRunner, TemporalStageRunner,
    GeneratorStageRunner, FusionStageRunner,
)

logger = get_logger(__name__)

STAGES = ["encoders", "temporal", "generator", "fusion", "all"]


def _get_domain_channels(cfg) -> dict[str, int]:
    """Count channels per domain from product config."""
    domain_ch: dict[str, int] = {}
    for product in cfg.data.products:
        d = product["domain"]
        if d == "meteo":
            continue
        domain_ch[d] = domain_ch.get(d, 0) + len(product["channels"])
    return domain_ch


def _get_domain_indices(cfg) -> dict[str, list[int]]:
    """Compute channel index ranges per domain."""
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


def train_encoders(cfg, output_dir, resume=None):
    logger.info("=== Stage 1: Training Spatial CNN Encoders ===")
    domain_ch = _get_domain_channels(cfg)
    domain_idx = _get_domain_indices(cfg)

    model = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    runner = EncoderStageRunner(
        model, train_loader, val_loader, cfg,
        output_dir / "encoders", domain_idx,
    )
    if resume:
        runner.load_checkpoint(resume)
    runner.fit()
    return model


def train_temporal(cfg, encoders, output_dir, resume=None):
    logger.info("=== Stage 2: Training Variational Transformer ===")
    domain_idx = _get_domain_indices(cfg)
    meteo_dim = len(domain_idx.get("meteo", []))

    model = VariationalTransformer(
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

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    runner = TemporalStageRunner(
        model, encoders, train_loader, val_loader, cfg,
        output_dir / "temporal", domain_idx,
    )
    if resume:
        runner.load_checkpoint(resume)
    runner.fit()
    return model


def train_generator(cfg, encoders, temporal_model, output_dir, resume=None):
    logger.info("=== Stage 3: Training Conditional UNet Generator ===")
    domain_idx = _get_domain_indices(cfg)
    total_ch = _total_channels(cfg)

    model = ConditionalUNet(
        state_dim=cfg.model.temporal.state_dim,
        out_channels=total_ch,
        noise_dim=cfg.model.generator.noise_dim,
        base_channels=cfg.model.generator.base_channels,
        channel_multipliers=list(cfg.model.generator.channel_multipliers),
        n_res_blocks=cfg.model.generator.n_res_blocks,
        use_spectral_norm=cfg.model.generator.use_spectral_norm,
        initial_spatial=tuple(cfg.model.generator.initial_spatial),
    )

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    runner = GeneratorStageRunner(
        model, encoders, temporal_model,
        train_loader, val_loader, cfg,
        output_dir / "generator", domain_idx,
    )
    if resume:
        runner.load_checkpoint(resume)
    runner.fit()
    return model


def train_fusion(cfg, encoders, temporal_model, generator, output_dir, resume=None):
    logger.info("=== Stage 4: Training Fusion Transformer ===")
    domain_idx = _get_domain_indices(cfg)
    total_ch = _total_channels(cfg)
    meteo_dim = len(domain_idx.get("meteo", []))

    model = FusionTransformer(
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

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    runner = FusionStageRunner(
        model, encoders, temporal_model, generator,
        train_loader, val_loader, cfg,
        output_dir / "fusion", domain_idx,
    )
    if resume:
        runner.load_checkpoint(resume)
    runner.fit()
    return model


@click.command()
@click.option("--stage", type=click.Choice(STAGES), required=True, help="Training stage")
@click.option("--config", "config_path", required=True, help="Path to config YAML")
@click.option("--override", multiple=True, help="Config overrides (key=value)")
@click.option("--resume", default=None, help="Path to checkpoint to resume from")
def main(stage, config_path, override, resume):
    # Load and validate config
    cfg = load_config(config_path, list(override) if override else None)
    validate_config(cfg)

    # Setup
    exp_id = cfg.project.get("experiment_id") or generate_experiment_id(cfg)
    output_dir = Path(cfg.project.output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")

    set_global_seed(cfg.project.seed)
    env_info = get_environment_info()
    logger.info(f"Experiment: {exp_id}")
    logger.info(f"Environment: {env_info}")

    if stage == "encoders" or stage == "all":
        encoders = train_encoders(cfg, output_dir, resume if stage == "encoders" else None)

    if stage == "temporal" or stage == "all":
        if stage == "temporal":
            # Load encoders from checkpoint
            domain_ch = _get_domain_channels(cfg)
            encoders = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
            enc_ckpt = output_dir / "encoders" / "checkpoints" / "best.ckpt"
            if enc_ckpt.exists():
                encoders.load_state_dict(torch.load(enc_ckpt, weights_only=False)["state_dict"])
        temporal_model = train_temporal(cfg, encoders, output_dir, resume if stage == "temporal" else None)

    if stage == "generator" or stage == "all":
        if stage == "generator":
            domain_ch = _get_domain_channels(cfg)
            encoders = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
            enc_ckpt = output_dir / "encoders" / "checkpoints" / "best.ckpt"
            if enc_ckpt.exists():
                encoders.load_state_dict(torch.load(enc_ckpt, weights_only=False)["state_dict"])

            domain_idx = _get_domain_indices(cfg)
            meteo_dim = len(domain_idx.get("meteo", []))
            temporal_model = VariationalTransformer(
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
            temp_ckpt = output_dir / "temporal" / "checkpoints" / "best.ckpt"
            if temp_ckpt.exists():
                temporal_model.load_state_dict(torch.load(temp_ckpt, weights_only=False)["state_dict"])

        generator = train_generator(
            cfg, encoders, temporal_model, output_dir,
            resume if stage == "generator" else None,
        )

    if stage == "fusion" or stage == "all":
        if stage == "fusion":
            # Load all previous models
            domain_ch = _get_domain_channels(cfg)
            domain_idx = _get_domain_indices(cfg)
            meteo_dim = len(domain_idx.get("meteo", []))
            total_ch = _total_channels(cfg)

            encoders = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
            temporal_model = VariationalTransformer(
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

            for name, model, subdir in [
                ("encoders", encoders, "encoders"),
                ("temporal", temporal_model, "temporal"),
                ("generator", generator, "generator"),
            ]:
                ckpt = output_dir / subdir / "checkpoints" / "best.ckpt"
                if ckpt.exists():
                    model.load_state_dict(torch.load(ckpt, weights_only=False)["state_dict"])
                    logger.info(f"Loaded {name} from {ckpt}")

        train_fusion(
            cfg, encoders, temporal_model, generator, output_dir,
            resume if stage == "fusion" else None,
        )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
