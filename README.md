# GOES-L2 Probabilistic Earth-System Forecasting Framework

Probabilistic atmospheric forecasting from GOES-16/17/18 Level 2 satellite data using a cascaded deep learning pipeline with uncertainty quantification.

```
surface + wind + humidity  →  future atmospheric state  →  reconstructed GOES L2 fields
```

## Architecture

The system trains in four sequential stages, each building on frozen outputs from the previous:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 · SPATIAL CNN ENCODERS                                 │
│  Three domain-specific CNNs (land, sea, cloud) with residual    │
│  blocks and CBAM attention. Each produces a latent embedding    │
│  and a next-step reconstruction.                                │
│  Loss: MSE + SSIM + PhysicsConstraint                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ latent embeddings [B, T, D]
┌────────────────────────────▼────────────────────────────────────┐
│  STAGE 2 · VARIATIONAL TRANSFORMER                              │
│  Fuses spatial latents with wind, humidity, temperature, and    │
│  pressure via a VAE bottleneck. Outputs μ and σ² per step.     │
│  Loss: ELBO (reconstruction + β·KL) + CRPS                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ predicted state distribution
┌────────────────────────────▼────────────────────────────────────┐
│  STAGE 3 · CONDITIONAL UNET (Reverse Generator)                 │
│  Maps predicted states back to synthetic GOES L2 fields via     │
│  FiLM conditioning at each decoder level.                       │
│  Loss: MSE + SSIM + SpectralLoss                                │
└────────────────────────────┬────────────────────────────────────┘
                             │ synthetic L2 fields
┌────────────────────────────▼────────────────────────────────────┐
│  STAGE 4 · FUSION TRANSFORMER                                   │
│  Combines all prediction streams into a unified holistic        │
│  forecast with spatial uncertainty maps.                        │
│  Loss: MSE + SSIM + CRPS + PhysicsConstraint                   │
└─────────────────────────────────────────────────────────────────┘
```

## GOES L2 Data Products

| Product | Description | Domain |
|---|---|---|
| ABI-L2-SSTF | Sea surface temperature | Sea encoder |
| ABI-L2-LSTF | Land surface temperature | Land encoder |
| ABI-L2-CMIPF | Cloud and moisture imagery (bands 7–16) | Cloud encoder |
| ABI-L2-DMWF | Derived motion winds | Meteo field |
| ABI-L2-TPWF | Total precipitable water | Meteo field |
| ABI-L2-AODF | Aerosol optical depth (optional) | Supplement |

## Installation

```bash
git clone <repo-url>
cd goes_forecast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"
```

Core: `torch>=2.2`, `xarray`, `zarr`, `scipy`, `omegaconf`, `click`
Optional: `goes2go`, `pyresample`, `pytorch-msssim`, `wandb`, `cmocean`

## Quick Start

### 1. Download data

```bash
python scripts/download_data.py \
  --satellite GOES-16 \
  --start 2023-01-01 --end 2023-12-31 \
  --products SST LST CMIP DMW TPW \
  --output-dir data/raw
```

### 2. Preprocess to Zarr

```bash
python scripts/preprocess.py \
  --input data/raw --output data/processed/goes_l2.zarr \
  --config config/base_config.yaml
```

### 3. Train

```bash
# Individual stages
python scripts/train.py --stage encoders --config config/base_config.yaml
python scripts/train.py --stage temporal --config config/base_config.yaml
python scripts/train.py --stage generator --config config/base_config.yaml
python scripts/train.py --stage fusion --config config/base_config.yaml

# Or all at once
python scripts/train.py --stage all --config config/base_config.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint outputs/fusion/checkpoints/best.ckpt \
  --split test --lead-times 1,3,6,12,24 --n-samples 50
```

## Configuration

All hyperparameters live in `config/base_config.yaml`. Override from CLI:

```bash
python scripts/train.py --stage encoders \
  --override model.encoder.latent_dim=256 training.stages.encoders.lr=2e-4
```

## Project Layout

```
goes_forecast/
├── config/              YAML configs
├── data/                Downloader, preprocessor, dataset, Zarr utilities
├── models/              blocks, spatial_encoder, temporal_bayesian,
│                        reverse_generator, fusion
├── training/            losses, trainer, stage_runners, callbacks
├── evaluation/          metrics, calibration, visualizer
├── utils/               config_loader, logger, reproducibility, projection
├── scripts/             download_data, preprocess, train, evaluate
└── tests/               pytest test suite
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Separate domain encoders | Different physical regimes benefit from specialized features |
| VAE bottleneck | Principled uncertainty; posterior collapse is diagnosable |
| FiLM conditioning | Stable state injection preserving spatial structure |
| Staged training | Prevents gradient conflicts; independent debugging |
| Zarr + Blosc | Cloud-native chunked format; efficient sequential access |
| Physics constraint losses | Steers toward physically plausible trajectories |

## Evaluation Metrics

| Metric | Type |
|---|---|
| RMSE / MAE / Bias | Deterministic accuracy |
| SSIM | Spatial structural quality |
| Spatial ACC | Anomaly correlation |
| CRPS | Probabilistic skill |
| Coverage (90% CI) | Calibration (target ≈ 0.9) |
| Reliability diagram | Full calibration curve |

## Running Tests

```bash
pytest tests/ -v --tb=short
```

## License

MIT
