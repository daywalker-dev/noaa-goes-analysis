"""
Smoke-test suite for the GOES L2 forecasting framework.

Run with:
    pytest tests/test_smoke.py -v

Requires: torch, numpy, scipy
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ===========================================================================
# Config
# ===========================================================================

def test_config_load_and_validate():
    # FIXED: was `from utils.config import ...`
    from utils.config_loader import load_config, validate_config
    cfg = load_config("config/default.yaml")
    validate_config(cfg)
    assert cfg.project.name == "goes_forecast"


def test_config_save_and_reload(tmp_path):
    from utils.config_loader import load_config, save_config
    cfg = load_config("config/default.yaml")
    out = tmp_path / "test_cfg.yaml"
    save_config(cfg, out)
    reloaded = load_config(str(out))
    assert reloaded.project.seed == cfg.project.seed


# ===========================================================================
# Reproducibility
# ===========================================================================

def test_seed_everything():
    # FIXED: was `from utils.reproducibility import seed_everything`
    from utils.reproducibility import set_global_seed
    set_global_seed(0)
    a = torch.randn(5)
    set_global_seed(0)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_get_environment_info():
    from utils.reproducibility import get_environment_info
    info = get_environment_info()
    assert "python" in info
    assert "torch" in info


# ===========================================================================
# Data
# ===========================================================================

def test_augmentation_random_flip():
    from data.augmentation import RandomFlip
    flip = RandomFlip(horizontal=True, vertical=True, p=1.0)
    x = torch.arange(16).reshape(1, 4, 4).float()
    out = flip(x)
    # At p=1.0 both flips always apply, so it's a 180° rotation
    assert out.shape == x.shape


def test_augmentation_compose():
    from data.augmentation import Compose, RandomFlip
    t = Compose([RandomFlip(p=0.0)])  # p=0 means no-op
    x = torch.randn(3, 8, 8)
    assert torch.allclose(t(x), x)


# ===========================================================================
# Models — smoke test forward passes
# ===========================================================================

def test_spatial_encoder_forward():
    from models.spatial_encoder import SpatialCNNEncoder
    model = SpatialCNNEncoder(in_channels=3, latent_dim=32, base_channels=16,
                               channel_multipliers=[1, 2])
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert "latent" in out
    assert "reconstruction" in out
    assert out["latent"].shape == (2, 32)


def test_domain_encoder_ensemble():
    from models.spatial_encoder import DomainEncoderEnsemble
    from types import SimpleNamespace
    cfg = SimpleNamespace(
        latent_dim=32, base_channels=16, channel_multipliers=[1, 2],
        n_res_blocks=2, use_cbam=True, dropout=0.0,
    )
    domain_ch = {"land": 2, "sea": 3}
    model = DomainEncoderEnsemble(domain_ch, cfg)
    inputs = {
        "land": torch.randn(2, 2, 32, 32),
        "sea": torch.randn(2, 3, 32, 32),
    }
    out = model(inputs)
    assert "latents" in out
    assert out["latents"].shape[0] == 2


def test_conditional_unet_forward():
    from models.reverse_generator import ConditionalUNet
    model = ConditionalUNet(
        state_dim=32, out_channels=3, noise_dim=8,
        base_channels=16, channel_multipliers=[4, 2, 1],
        initial_spatial=(8, 8),
    )
    state = torch.randn(2, 32)
    out = model(state, target_size=(64, 64))
    assert out.shape == (2, 3, 64, 64)


# ===========================================================================
# Losses
# ===========================================================================

def test_masked_mse_perfect():
    from training.losses import MaskedMSE
    loss = MaskedMSE()
    x = torch.randn(4, 3, 16, 16)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_crps_loss_zero_variance():
    from training.losses import CRPSLoss
    loss = CRPSLoss()
    import math
    mu = torch.tensor([1.0, 2.0])
    logvar = torch.full((2,), math.log(0.01))
    target = torch.tensor([1.0, 2.0])
    val = loss(mu, logvar, target)
    assert val.item() < 0.1


# ===========================================================================
# Evaluation
# ===========================================================================

def test_rmse_zero_error():
    from evaluation.metrics import rmse
    pred = np.ones((10, 10))
    assert rmse(pred, pred) == pytest.approx(0.0, abs=1e-8)


def test_spatial_correlation_perfect():
    from evaluation.metrics import spatial_correlation
    x = np.random.randn(3, 16, 16)
    corr = spatial_correlation(x, x)
    assert corr == pytest.approx(1.0, abs=1e-5)