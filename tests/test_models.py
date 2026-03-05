"""Unit tests for all model architectures: shape correctness and forward passes."""
import pytest
import torch
from types import SimpleNamespace

from models.spatial_encoder import SpatialCNNEncoder, DomainEncoderEnsemble
from models.temporal_bayesian import VariationalTransformer
from models.reverse_generator import ConditionalUNet
from models.fusion import FusionTransformer


def _make_encoder_cfg(**overrides):
    defaults = dict(
        latent_dim=64, base_channels=16, channel_multipliers=[1, 2, 4],
        n_res_blocks=2, use_cbam=True, dropout=0.0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSpatialCNNEncoder:
    def test_forward_shape(self, device):
        model = SpatialCNNEncoder(
            in_channels=3, latent_dim=64, base_channels=16,
            channel_multipliers=[1, 2, 4], n_res_blocks=2,
        ).to(device)
        x = torch.randn(2, 3, 64, 64, device=device)
        out = model(x)
        assert out["latent"].shape == (2, 64)
        assert out["reconstruction"].shape == (2, 3, 64, 64)

    def test_encode_only(self, device):
        model = SpatialCNNEncoder(
            in_channels=2, latent_dim=32, base_channels=16,
            channel_multipliers=[1, 2],
        ).to(device)
        x = torch.randn(2, 2, 32, 32, device=device)
        z = model.encode_only(x)
        assert z.shape == (2, 32)

    def test_gradient_flow(self, device):
        model = SpatialCNNEncoder(
            in_channels=2, latent_dim=32, base_channels=16,
            channel_multipliers=[1, 2],
        ).to(device)
        x = torch.randn(1, 2, 32, 32, device=device, requires_grad=True)
        out = model(x)
        (out["latent"].sum() + out["reconstruction"].sum()).backward()
        assert x.grad is not None


class TestDomainEncoderEnsemble:
    def test_forward(self, device):
        domain_ch = {"land": 2, "sea": 3}
        cfg = _make_encoder_cfg(latent_dim=32, channel_multipliers=[1, 2])
        model = DomainEncoderEnsemble(domain_ch, cfg).to(device)

        inputs = {
            "land": torch.randn(2, 2, 32, 32, device=device),
            "sea": torch.randn(2, 3, 32, 32, device=device),
        }
        results = model(inputs)
        assert results["latents"].shape == (2, 64)  # 2 domains × 32
        assert results["latent_land"].shape == (2, 32)
        assert results["recon_sea"].shape == (2, 3, 32, 32)

    def test_encode_only(self, device):
        domain_ch = {"land": 1, "sea": 1}
        cfg = _make_encoder_cfg(latent_dim=16, channel_multipliers=[1, 2])
        model = DomainEncoderEnsemble(domain_ch, cfg).to(device)

        inputs = {
            "land": torch.randn(2, 1, 32, 32, device=device),
            "sea": torch.randn(2, 1, 32, 32, device=device),
        }
        z = model.encode_only(inputs)
        assert z.shape == (2, 32)  # 2 × 16

    def test_freeze_unfreeze(self, device):
        domain_ch = {"land": 2}
        cfg = _make_encoder_cfg(latent_dim=16, channel_multipliers=[1, 2])
        model = DomainEncoderEnsemble(domain_ch, cfg).to(device)

        model.freeze()
        assert all(not p.requires_grad for p in model.parameters())
        model.unfreeze()
        assert all(p.requires_grad for p in model.parameters())


class TestVariationalTransformer:
    def test_forward_shape(self, device):
        model = VariationalTransformer(
            latent_dim=64, meteo_dim=4, state_dim=32,
            d_model=32, n_heads=2, n_encoder_layers=1, n_decoder_layers=1,
            dim_feedforward=64, forecast_steps=6,
        ).to(device)
        latents = torch.randn(2, 8, 64, device=device)
        meteo = torch.randn(2, 8, 4, device=device)
        out = model(latents, meteo)
        assert out["mu"].shape == (2, 6, 32)
        assert out["logvar"].shape == (2, 6, 32)
        assert out["kl_loss"].ndim == 0  # scalar

    def test_kl_positive(self, device):
        model = VariationalTransformer(
            latent_dim=32, meteo_dim=2, state_dim=16,
            d_model=16, n_heads=2, n_encoder_layers=1, n_decoder_layers=1,
            dim_feedforward=32, forecast_steps=4,
        ).to(device)
        model.train()
        out = model(torch.randn(2, 4, 32, device=device), torch.randn(2, 4, 2, device=device))
        assert out["kl_loss"].item() >= 0

    def test_sample(self, device):
        model = VariationalTransformer(
            latent_dim=32, meteo_dim=2, state_dim=16,
            d_model=16, n_heads=2, n_encoder_layers=1, n_decoder_layers=1,
            dim_feedforward=32, forecast_steps=4,
        ).to(device)
        out = model.sample(
            torch.randn(2, 4, 32, device=device),
            torch.randn(2, 4, 2, device=device),
            n_samples=5,
        )
        assert out["mean"].shape == (2, 4, 16)
        assert out["std"].shape == (2, 4, 16)
        assert out["p05"].shape == (2, 4, 16)
        assert out["p95"].shape == (2, 4, 16)


class TestConditionalUNet:
    def test_forward_shape(self, device):
        model = ConditionalUNet(
            state_dim=64, out_channels=4, noise_dim=8,
            base_channels=16, channel_multipliers=[4, 2, 1],
            initial_spatial=(8, 8),
        ).to(device)
        state = torch.randn(2, 64, device=device)
        out = model(state, target_size=(64, 64))
        assert out.shape == (2, 4, 64, 64)

    def test_decode_sequence(self, device):
        model = ConditionalUNet(
            state_dim=32, out_channels=3, noise_dim=8,
            base_channels=16, channel_multipliers=[4, 2, 1],
            initial_spatial=(8, 8),
        ).to(device)
        states = torch.randn(2, 6, 32, device=device)
        out = model.decode_sequence(states, target_size=(64, 64))
        assert out.shape == (2, 6, 3, 64, 64)

    def test_gradient_flow(self, device):
        model = ConditionalUNet(
            state_dim=32, out_channels=2, noise_dim=4,
            base_channels=8, channel_multipliers=[2, 1],
            initial_spatial=(4, 4),
        ).to(device)
        state = torch.randn(1, 32, device=device, requires_grad=True)
        out = model(state)
        out.sum().backward()
        assert state.grad is not None


class TestFusionTransformer:
    def test_forward_shape(self, device):
        model = FusionTransformer(
            cnn_dim=64, bayes_dim=64, meteo_dim=4, gen_dim=8,
            d_model=32, n_heads=2, n_layers=1, dim_feedforward=64,
            out_channels=4, spatial_size=(16, 16),
        ).to(device)
        B, T = 2, 6
        gen_spatial = torch.randn(B, T, 4, 16, 16, device=device)
        out = model(
            cnn_latents=torch.randn(B, T, 64, device=device),
            bayes_output=torch.randn(B, T, 64, device=device),
            meteo_fields=torch.randn(B, T, 4, device=device),
            gen_features=torch.randn(B, T, 8, device=device),
            gen_spatial=gen_spatial,
        )
        assert out["scales"].shape == (B, T, 4)
        assert out["uncertainty"].shape == (B, T, 4)
        assert "forecast" in out
        assert out["forecast"].shape == (B, T, 4, 16, 16)

    def test_without_spatial(self, device):
        model = FusionTransformer(
            cnn_dim=32, bayes_dim=32, meteo_dim=2, gen_dim=4,
            d_model=16, n_heads=2, n_layers=1, dim_feedforward=32,
            out_channels=2,
        ).to(device)
        B, T = 1, 3
        out = model(
            cnn_latents=torch.randn(B, T, 32, device=device),
            bayes_output=torch.randn(B, T, 32, device=device),
            meteo_fields=torch.randn(B, T, 2, device=device),
            gen_features=torch.randn(B, T, 4, device=device),
        )
        assert "scales" in out
        assert "forecast" not in out  # No gen_spatial provided
