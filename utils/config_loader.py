"""YAML configuration loader with OmegaConf and CLI override support."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel


class EncoderConfig(BaseModel):
    latent_dim: int = 128
    n_res_blocks: int = 4
    base_channels: int = 64
    channel_multipliers: list[int] = [1, 2, 4, 8]
    use_cbam: bool = True
    dropout: float = 0.1


class TemporalConfig(BaseModel):
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    meteo_dim: int = 64
    state_dim: int = 256
    beta_kl: float = 1.0
    free_bits: float = 0.5
    kl_anneal_epochs: int = 20


class GeneratorConfig(BaseModel):
    base_channels: int = 64
    channel_multipliers: list[int] = [8, 4, 2, 1]
    n_res_blocks: int = 2
    noise_dim: int = 32
    use_spectral_norm: bool = True
    initial_spatial: list[int] = [8, 8]


class FusionModelConfig(BaseModel):
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    n_sources: int = 4


def load_config(
    config_path: str | Path,
    overrides: Optional[list[str]] = None,
) -> DictConfig:
    """Load YAML config with optional CLI overrides."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """Validate config sections with Pydantic schemas."""
    m = OmegaConf.to_container(cfg.model, resolve=True)
    EncoderConfig(**m["encoder"])
    TemporalConfig(**m["temporal"])
    GeneratorConfig(**m["generator"])
    FusionModelConfig(**m["fusion"])


def generate_experiment_id(cfg: DictConfig) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha256(OmegaConf.to_yaml(cfg).encode()).hexdigest()[:8]
    return f"run_{ts}_{h}"


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def get_stage_config(cfg: DictConfig, stage: str) -> DictConfig:
    if stage not in cfg.training.stages:
        raise KeyError(f"Unknown stage '{stage}'. Available: {list(cfg.training.stages.keys())}")
    return cfg.training.stages[stage]
