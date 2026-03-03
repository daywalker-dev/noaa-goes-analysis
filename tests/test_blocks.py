"""Unit tests for model building blocks: shape correctness and gradient flow."""
import pytest
import torch

from goes_forecast.models.blocks import (
    CBAM, ChannelAttention, DownBlock, FiLM,
    ResidualBlock, SinusoidalPE, SpatialAttention, UpBlock,
)


class TestResidualBlock:
    def test_shape(self, device):
        block = ResidualBlock(64).to(device)
        x = torch.randn(2, 64, 32, 32, device=device)
        assert block(x).shape == x.shape

    def test_gradient_flow(self, device):
        block = ResidualBlock(32).to(device)
        x = torch.randn(1, 32, 16, 16, device=device, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None


class TestChannelAttention:
    def test_shape(self, device):
        ca = ChannelAttention(64).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        assert ca(x).shape == x.shape


class TestSpatialAttention:
    def test_shape(self, device):
        sa = SpatialAttention().to(device)
        x = torch.randn(2, 32, 16, 16, device=device)
        assert sa(x).shape == x.shape


class TestCBAM:
    def test_shape(self, device):
        cbam = CBAM(64).to(device)
        x = torch.randn(2, 64, 16, 16, device=device)
        assert cbam(x).shape == x.shape

    def test_gradient_flow(self, device):
        cbam = CBAM(32).to(device)
        x = torch.randn(1, 32, 8, 8, device=device, requires_grad=True)
        cbam(x).sum().backward()
        assert x.grad is not None


class TestFiLM:
    def test_shape(self, device):
        film = FiLM(cond_dim=64, channels=32).to(device)
        x = torch.randn(2, 32, 16, 16, device=device)
        cond = torch.randn(2, 64, device=device)
        assert film(x, cond).shape == x.shape


class TestSinusoidalPE:
    def test_shape(self, device):
        pe = SinusoidalPE(128).to(device)
        x = torch.randn(2, 10, 128, device=device)
        assert pe(x).shape == x.shape

    def test_different_from_input(self, device):
        pe = SinusoidalPE(64).to(device)
        x = torch.zeros(1, 5, 64, device=device)
        out = pe(x)
        assert not torch.allclose(out, x)  # PE should modify zeros


class TestDownBlock:
    def test_shape(self, device):
        block = DownBlock(32, 64).to(device)
        x = torch.randn(2, 32, 32, 32, device=device)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)  # spatial halved


class TestUpBlock:
    def test_shape_no_skip(self, device):
        block = UpBlock(64, 32).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = block(x)
        assert out.shape == (2, 32, 16, 16)  # spatial doubled

    def test_shape_with_skip(self, device):
        block = UpBlock(64, 32).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        skip = torch.randn(2, 32, 16, 16, device=device)
        out = block(x, skip)
        assert out.shape == (2, 32, 16, 16)
