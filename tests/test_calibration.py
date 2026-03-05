"""Tests for config loading and calibration analysis."""
import numpy as np
import pytest

from evaluation.calibration import (
    reliability_diagram, rank_histogram, sharpness, calibration_summary,
)


class TestReliabilityDiagram:
    def test_perfect_calibration(self):
        # Large sample with known Gaussian → reliability should be close to diagonal
        np.random.seed(42)
        n = 10000
        mu = np.zeros(n)
        sigma = np.ones(n)
        target = np.random.randn(n)  # samples from N(0,1)
        result = reliability_diagram(mu, sigma, target, n_bins=10)
        assert result["ice"] < 0.05  # should be well calibrated
        assert len(result["nominal"]) == 10
        assert len(result["observed"]) == 10

    def test_overconfident(self):
        # Sigma too small → observations fall outside → observed < nominal
        np.random.seed(42)
        n = 5000
        mu = np.zeros(n)
        sigma = np.ones(n) * 0.01  # way too narrow
        target = np.random.randn(n)
        result = reliability_diagram(mu, sigma, target)
        assert result["ice"] > 0.1  # poorly calibrated


class TestRankHistogram:
    def test_shape(self):
        np.random.seed(42)
        N = 20  # ensemble size
        M = 100
        samples = np.random.randn(N, M)
        target = np.random.randn(M)
        hist = rank_histogram(samples, target)
        assert hist.shape == (N + 1,)
        assert hist.sum() == M

    def test_uniform_ensemble(self):
        # Well-calibrated ensemble → approximately flat histogram
        np.random.seed(42)
        N = 50
        M = 5000
        mu = np.random.randn(M)
        samples = mu[None, :] + np.random.randn(N, M)
        target = mu + np.random.randn(M)
        hist = rank_histogram(samples, target)
        # Coefficient of variation should be small for flat histogram
        cv = np.std(hist) / np.mean(hist)
        assert cv < 0.5  # reasonably flat


class TestSharpness:
    def test_known_value(self):
        sigma = np.ones(100)
        # 80% CI → z ≈ 1.282 → width = 2 * 1.282 * 1 ≈ 2.564
        s = sharpness(sigma, confidence=0.8)
        assert s == pytest.approx(2.563, abs=0.01)

    def test_narrow_vs_wide(self):
        narrow = sharpness(np.ones(100) * 0.1)
        wide = sharpness(np.ones(100) * 10.0)
        assert narrow < wide


class TestCalibrationSummary:
    def test_keys(self):
        np.random.seed(42)
        mu = np.zeros(1000)
        sigma = np.ones(1000)
        target = np.random.randn(1000)
        summary = calibration_summary(mu, sigma, target)
        assert "50%" in summary
        assert "90%" in summary
        assert "actual" in summary["90%"]
        assert "nominal" in summary["90%"]
        assert "gap" in summary["90%"]
