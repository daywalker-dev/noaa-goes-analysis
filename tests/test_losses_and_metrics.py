"""Tests for loss functions and evaluation metrics."""
import math

import numpy as np
import pytest
import torch

from goes_forecast.training.losses import (
    CRPSLoss, ELBOLoss, MaskedMSE, PhysicsConstraintLoss,
    SSIMLoss, SpectralLoss,
)
from goes_forecast.evaluation.metrics import (
    bias, coverage_score, crps_gaussian, mae, rmse,
    spatial_correlation, ssim_score,
)


# ---- Loss function tests ----

class TestMaskedMSE:
    def test_perfect_prediction(self):
        loss = MaskedMSE()
        x = torch.randn(4, 3, 16, 16)
        assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)

    def test_masked(self):
        loss = MaskedMSE()
        pred = torch.ones(2, 1, 4, 4)
        target = torch.zeros(2, 1, 4, 4)
        mask = torch.zeros(2, 1, 4, 4)
        mask[:, :, :2, :2] = 1.0
        result = loss(pred, target, mask)
        assert result.item() == pytest.approx(1.0, abs=1e-6)

    def test_no_mask(self):
        loss = MaskedMSE()
        pred = torch.ones(2, 1, 4, 4) * 2
        target = torch.zeros(2, 1, 4, 4)
        assert loss(pred, target).item() == pytest.approx(4.0, abs=1e-6)


class TestSSIMLoss:
    def test_identical(self):
        loss = SSIMLoss()
        x = torch.randn(2, 1, 32, 32)
        val = loss(x, x)
        assert val.item() == pytest.approx(0.0, abs=0.05)  # 1 - SSIM ≈ 0

    def test_different(self):
        loss = SSIMLoss()
        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 1, 32, 32)
        val = loss(x, y)
        assert val.item() > 0.0  # Should be > 0 for different inputs


class TestCRPSLoss:
    def test_perfect_gaussian(self):
        loss = CRPSLoss()
        mu = torch.tensor([1.0, 2.0, 3.0])
        logvar = torch.full((3,), math.log(0.01))  # very small variance
        target = torch.tensor([1.0, 2.0, 3.0])
        val = loss(mu, logvar, target)
        assert val.item() < 0.1

    def test_large_error(self):
        loss = CRPSLoss()
        mu = torch.zeros(100)
        logvar = torch.zeros(100)
        target = torch.ones(100) * 10
        val = loss(mu, logvar, target)
        assert val.item() > 1.0


class TestSpectralLoss:
    def test_identical(self):
        loss = SpectralLoss()
        x = torch.randn(2, 1, 32, 32)
        assert loss(x, x).item() == pytest.approx(0.0, abs=1e-5)

    def test_different(self):
        loss = SpectralLoss()
        x = torch.randn(2, 1, 32, 32)
        y = torch.randn(2, 1, 32, 32)
        assert loss(x, y).item() > 0


class TestPhysicsConstraintLoss:
    def test_smooth_sequence(self):
        loss = PhysicsConstraintLoss()
        # Constant sequence should have low temporal smoothness loss
        x = torch.ones(2, 4, 1, 16, 16)
        val = loss(x, x)
        assert val.item() >= 0

    def test_with_targets(self):
        loss = PhysicsConstraintLoss()
        pred = torch.randn(2, 4, 1, 16, 16)
        target = torch.randn(2, 4, 1, 16, 16)
        val = loss(pred, target)
        assert val.item() >= 0


class TestELBOLoss:
    def test_components(self):
        elbo = ELBOLoss(beta=1.0, free_bits=0.5)
        mu = torch.randn(4, 10)
        logvar = torch.zeros(4, 10)
        target = mu.clone()
        out = elbo(mu, logvar, target)
        assert "total" in out
        assert "recon" in out
        assert "kl" in out
        assert out["recon"].item() < 0.01  # perfect recon

    def test_kl_positive(self):
        elbo = ELBOLoss(beta=1.0, free_bits=0.0)
        mu = torch.randn(4, 10) * 2  # non-zero mean
        logvar = torch.ones(4, 10)   # non-unit variance
        target = torch.zeros(4, 10)
        out = elbo(mu, logvar, target)
        assert out["kl"].item() > 0

    def test_beta_annealing(self):
        elbo = ELBOLoss(beta=0.0)
        mu = torch.randn(4, 10)
        logvar = torch.ones(4, 10)
        out_beta0 = elbo(mu, logvar, mu)
        elbo.set_beta(1.0)
        out_beta1 = elbo(mu, logvar, mu)
        assert out_beta1["total"].item() >= out_beta0["total"].item()


# ---- Evaluation metrics tests ----

class TestEvalMetrics:
    def test_rmse_perfect(self):
        x = np.ones((10, 10))
        assert rmse(x, x) == pytest.approx(0.0)

    def test_rmse_known(self):
        pred = np.ones(100)
        target = np.zeros(100)
        assert rmse(pred, target) == pytest.approx(1.0)

    def test_mae_perfect(self):
        x = np.random.randn(10, 10)
        assert mae(x, x) == pytest.approx(0.0)

    def test_bias(self):
        pred = np.ones(100)
        target = np.zeros(100)
        assert bias(pred, target) == pytest.approx(1.0)

    def test_bias_negative(self):
        pred = np.zeros(100)
        target = np.ones(100)
        assert bias(pred, target) == pytest.approx(-1.0)

    def test_spatial_correlation_perfect(self):
        x = np.random.randn(50, 50)
        assert spatial_correlation(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_spatial_correlation_anticorrelated(self):
        x = np.random.randn(50, 50)
        assert spatial_correlation(x, -x) == pytest.approx(-1.0, abs=1e-6)

    def test_ssim_identical(self):
        x = np.random.randn(32, 32)
        assert ssim_score(x, x) > 0.99

    def test_crps_gaussian_perfect(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.01, 0.01, 0.01])
        obs = np.array([1.0, 2.0, 3.0])
        assert crps_gaussian(mu, sigma, obs) < 0.01

    def test_crps_gaussian_imperfect(self):
        mu = np.zeros(100)
        sigma = np.ones(100)
        obs = np.ones(100) * 5
        assert crps_gaussian(mu, sigma, obs) > 0.5

    def test_coverage_all_inside(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([10.0, 10.0, 10.0])  # wide CI
        target = np.array([1.0, 2.0, 3.0])
        assert coverage_score(mu, sigma, target, confidence=0.9) == pytest.approx(1.0)

    def test_coverage_none_inside(self):
        mu = np.array([0.0, 0.0])
        sigma = np.array([0.001, 0.001])  # very narrow CI
        target = np.array([100.0, 100.0])
        assert coverage_score(mu, sigma, target, confidence=0.9) == pytest.approx(0.0)
