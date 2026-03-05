"""
Smoke-test suite for the GOES L2 forecasting framework.

Run with:
    pytest goes_forecast/tests/test_smoke.py -v

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
    from utils.config import load_config, validate_config
    cfg = load_config("configs/default.yaml")
    validate_config(cfg)
    assert cfg["project"]["name"] == "goes_l2_forecast"


def test_config_merge():
    from utils.config import load_config, merge_configs, get_nested
    base = load_config("configs/default.yaml")
    override = {"project": {"seed": 999}, "temporal": {"type": "transformer"}}
    merged = merge_configs(base, override)
    assert get_nested(merged, "project.seed") == 999
    assert get_nested(merged, "temporal.type") == "transformer"
    # Base unchanged
    assert get_nested(base, "project.seed") == 42


# ===========================================================================
# Reproducibility
# ===========================================================================

def test_seed_everything():
    from utils.reproducibility import seed_everything
    seed_everything(0)
    a = torch.randn(5)
    seed_everything(0)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_early_stopping():
    from utils.reproducibility import EarlyStopping
    es = EarlyStopping(patience=3)
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.95)  # no improvement but patience=3
    assert not es.step(0.96)
    assert not es.step(0.97)
    assert es.step(0.98)      # patience exhausted


# ===========================================================================
# Data
# ===========================================================================

def test_field_normalizer_roundtrip():
    from data.ingest import FieldNormalizer
    data = np.random.randn(10, 3, 8, 8).astype(np.float32) * 50 + 200
    names = ["A", "B", "C"]

    for method in ["minmax", "zscore", "robust"]:
        norm = FieldNormalizer(method)
        norm.fit(data, names)
        normed = norm.transform(data, names)
        inv = norm.inverse_transform(normed, names)
        assert np.allclose(data, inv, atol=1e-4), f"Roundtrip failed for {method}"


def test_fill_missing():
    from data.ingest import fill_missing
    data = np.random.randn(5, 1, 4, 4).astype(np.float32)
    data[0, 0, 0, 0] = np.nan
    filled, mask = fill_missing(data, "zero")
    assert np.isfinite(filled).all()
    assert not mask[0, 0, 0, 0]


def test_dataset_shapes():
    from data.dataset import GOESL2Dataset
    T, H, W = 40, 16, 16
    arrays = {
        "LST": np.random.randn(T, H, W).astype(np.float32),
        "SST": np.random.randn(T, H, W).astype(np.float32),
        "CMI": np.random.randn(T, H, W).astype(np.float32),
        "TPW": np.random.randn(T, H, W).astype(np.float32),
        "wind_speed": np.random.randn(T, H, W).astype(np.float32),
        "wind_direction": np.random.randn(T, H, W).astype(np.float32),
    }
    masks = {k: np.ones((T, H, W), dtype=bool) for k in arrays}
    ds = GOESL2Dataset(arrays, masks, window_size=8, forecast_horizon=4)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["land_in"].shape[0] == 8     # window
    assert sample["land_target"].shape[0] == 4  # horizon


# ===========================================================================
# Models
# ===========================================================================

def test_residual_block():
    from models.blocks import ResidualBlock
    blk = ResidualBlock(32)
    x = torch.randn(2, 32, 16, 16)
    y = blk(x)
    assert y.shape == x.shape


def test_spatial_encoder_forward():
    from models.encoders.spatial_cnn import SpatialEncoder
    enc = SpatialEncoder(in_channels=3, latent_dim=64, base_filters=16,
                         num_res_blocks=2, attention_layers=[1])
    x = torch.randn(2, 3, 32, 32)
    z, pred, feat = enc(x)
    assert z.shape == (2, 64)
    assert pred.shape[0] == 2
    assert pred.shape[1] == 3  # same as in_channels


def test_variational_rnn_forward():
    from models.temporal.probabilistic import VariationalRNN
    model = VariationalRNN(input_dim=64, hidden_dim=32, latent_dim=16, num_layers=1)
    x = torch.randn(2, 10, 64)
    out = model(x)
    assert out["pred_mu"].shape == (2, 10, 64)
    assert out["pred_logvar"].shape == (2, 10, 64)
    assert out["kl_loss"].ndim == 0


def test_variational_rnn_forecast():
    from models.temporal.probabilistic import VariationalRNN
    model = VariationalRNN(input_dim=64, hidden_dim=32, latent_dim=16, num_layers=1)
    model.eval()
    x = torch.randn(2, 10, 64)
    out = model(x)
    fc = model.forecast(x[:, -1], out["hidden"], steps=4, num_samples=5)
    assert fc["mean"].shape == (2, 4, 64)
    assert fc["std"].shape == (2, 4, 64)
    assert fc["samples"].shape == (5, 2, 4, 64)


def test_bayesian_lstm():
    from models.temporal.probabilistic import BayesianLSTM
    model = BayesianLSTM(input_dim=64, hidden_dim=32, latent_dim=16, num_layers=1)
    x = torch.randn(2, 10, 64)
    out = model(x)
    assert out["pred_mu"].shape == (2, 10, 64)


def test_transformer_uncertainty():
    from models.temporal.probabilistic import UncertaintyTransformer
    model = UncertaintyTransformer(input_dim=64, hidden_dim=32, latent_dim=16,
                                    num_layers=1, num_heads=4)
    x = torch.randn(2, 10, 64)
    out = model(x)
    assert out["pred_mu"].shape == (2, 10, 64)


def test_conditional_unet_decoder():
    from models.decoder.reverse_generator import ConditionalUNetDecoder
    dec = ConditionalUNetDecoder(in_channels=64, out_channels=6, base_filters=16,
                                  depth=2, target_size=(32, 32))
    z = torch.randn(2, 64)
    out = dec(z)
    assert out.shape == (2, 6, 32, 32)


def test_cross_attention_fusion():
    from models.fusion.fusion_model import CrossAttentionFusion
    model = CrossAttentionFusion(input_sources=4, hidden_dim=32, num_heads=4,
                                  num_layers=1, output_dim=6, spatial_size=(32, 32))
    sources = [torch.randn(2, 32) for _ in range(4)]
    out = model(sources)
    assert out.shape == (2, 6, 32, 32)


# ===========================================================================
# Losses
# ===========================================================================

def test_ssim_loss():
    from models.blocks import ssim_loss
    x = torch.randn(2, 3, 32, 32)
    loss = ssim_loss(x, x)
    assert loss.item() < 0.01  # near-zero for identical inputs


def test_combined_loss():
    from models.blocks import CombinedLoss
    criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.3, physics_weight=0.1)
    pred = torch.randn(2, 3, 32, 32)
    target = pred + torch.randn_like(pred) * 0.1
    loss = criterion(pred, target)
    assert loss.item() > 0


# ===========================================================================
# Evaluation
# ===========================================================================

def test_rmse_mae():
    from evaluation.metrics import rmse, mae
    pred = np.ones((10, 10))
    target = np.zeros((10, 10))
    assert rmse(pred, target) == pytest.approx(1.0)
    assert mae(pred, target) == pytest.approx(1.0)


def test_spatial_correlation():
    from evaluation.metrics import spatial_correlation
    x = np.random.randn(32, 32)
    assert spatial_correlation(x, x) == pytest.approx(1.0, abs=1e-6)
    assert spatial_correlation(x, -x) == pytest.approx(-1.0, abs=1e-6)


def test_crps_gaussian():
    from evaluation.metrics import crps_gaussian
    mu = np.zeros(100)
    sigma = np.ones(100)
    obs = np.zeros(100)
    score = crps_gaussian(mu, sigma, obs)
    assert 0 < score < 1  # should be ~0.23 for perfect mean, unit var


def test_run_evaluation():
    from evaluation.metrics import run_evaluation
    cfg = {"evaluation": {"metrics": ["rmse", "mae"]}}
    pred = {"mean": np.random.randn(5, 3, 16, 16).astype(np.float32)}
    tgt = {"obs": pred["mean"] + np.random.randn(*pred["mean"].shape).astype(np.float32) * 0.1}
    results = run_evaluation(pred, tgt, cfg)
    assert "rmse" in results
    assert "mae" in results


# ===========================================================================
# Temporal model factory
# ===========================================================================

@pytest.mark.parametrize("model_type", ["variational_rnn", "bayesian_lstm", "transformer"])
def test_temporal_factory(model_type):
    from models.temporal.probabilistic import build_temporal_model
    cfg = {
        "type": model_type,
        "input_dim": 64,
        "hidden_dim": 32,
        "latent_dim": 16,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.1,
    }
    model = build_temporal_model(cfg)
    x = torch.randn(2, 5, 64)
    out = model(x)
    assert "pred_mu" in out


# ===========================================================================
# Fusion model factory
# ===========================================================================

@pytest.mark.parametrize("fusion_type", ["cross_attention", "concat_mlp", "film"])
def test_fusion_factory(fusion_type):
    from models.fusion.fusion_model import build_fusion_model
    cfg = {
        "type": fusion_type,
        "input_sources": 3,
        "hidden_dim": 32,
        "num_heads": 4,
        "num_layers": 1,
        "output_dim": 4,
        "dropout": 0.1,
    }
    model = build_fusion_model(cfg, grid_size=(16, 16))
    sources = [torch.randn(2, 32) for _ in range(3)]
    out = model(sources)
    assert out.shape == (2, 4, 16, 16)