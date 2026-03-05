# GOES L2 Probabilistic Earth-System Forecasting

A modular, research-grade machine learning framework that learns surface and sea dynamics from GOES Level-2 satellite data, produces multi-step probabilistic forecasts with uncertainty quantification, and reconstructs synthetic satellite-style fields from predicted atmospheric states.

---

## Architecture Overview

The system is a four-stage pipeline where each stage trains a distinct model component:

```
                    ┌──────────────────────────────────────────┐
                    │          GOES L2 Products (Input)         │
                    │  SST · LST · CMI · DMW · TPW · Aerosol   │
                    └──────────┬───────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
    │  Land Encoder │  │  Sea Encoder  │  │ Cloud Encoder │
    │  (CNN + Attn) │  │  (CNN + Attn) │  │  (CNN + Attn) │
    └──────┬────────┘  └──────┬────────┘  └──────┬────────┘
           │ z_land           │ z_sea            │ z_cloud
           └──────────────────┼──────────────────┘
                              │  concat + wind + humidity + temp
                              ▼
                   ┌─────────────────────┐
                   │  Temporal Model     │
                   │  (Variational RNN / │
                   │   Bayesian LSTM /   │
                   │   Transformer)      │
                   │                     │
                   │  → mean + variance  │
                   │  → multi-step fcast │
                   └──────────┬──────────┘
                              │ z_atmos (predicted)
                              ▼
                   ┌─────────────────────┐
                   │  Reverse Generator  │
                   │  (Conditional UNet) │
                   │                     │
                   │  z_atmos → L2 maps  │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │    Fusion Model     │
                   │  (Cross-Attention)  │
                   │                     │
                   │  orig L2 + CNN pred │
                   │  + Bayesian out     │
                   │  + wind/humidity    │
                   │  → holistic fcast   │
                   └─────────────────────┘
```

### Stage 1 — Spatial CNN Encoders

Three separate CNN encoders (land, sea, cloud) process multi-channel L2 inputs. Each uses residual blocks with GroupNorm, GELU activations, and optional spatial self-attention at configurable layers. Each encoder produces a flat latent embedding and a next-step spatial prediction. Trained with MSE + SSIM + optional physics-aware loss.

### Stage 2 — Temporal Probabilistic Model

Latent embeddings from the encoders are concatenated with auxiliary atmospheric drivers (wind, humidity, temperature) and fed to a temporal model that outputs distributions over future atmospheric states. Three backends are available: Variational RNN (default), Bayesian LSTM (MC-Dropout), and causal Transformer with uncertainty heads. Produces mean + variance at each step and supports autoregressive multi-step forecasting.

### Stage 3 — Reverse Generator

A conditional UNet decoder maps predicted atmospheric-state vectors back to spatial fields that resemble GOES L2 products. Uses transposed convolutions, skip connections, and spatial self-attention at the bottleneck.

### Stage 4 — Fusion Model

Combines four information streams (original L2 data, CNN spatial predictions, temporal model outputs, wind/humidity features) via multi-head cross-attention to produce a unified holistic atmospheric forecast across all L2-derived fields.

---

## Data Flow

```
GOES L2 products (via goes2go)
  → Download hourly observations
  → Reproject to common regular grid (256×256 default)
  → Temporal alignment to uniform time axis
  → Channel normalization (min-max / z-score / robust)
  → Missing-data masking and fill
  → Store as Zarr (or NetCDF)
  → Sliding-window PyTorch Dataset
  → DataLoaders (temporal train/val split)
```

### Supported L2 Products

| Config Key      | Product                     | Variables         |
|-----------------|-----------------------------|-------------------|
| ABI-L2-SSTP     | Sea Surface Temperature     | SST               |
| ABI-L2-LSTP     | Land Surface Temperature    | LST               |
| ABI-L2-MCMIPP   | Cloud/Moisture Imagery      | CMI               |
| ABI-L2-DMWP     | Derived Motion Winds        | wind_speed, dir   |
| ABI-L2-TPWP     | Total Precipitable Water    | TPW               |
| ABI-L2-ADPP     | Aerosol Detection           | Aerosol           |

---

## Project Structure

```
goes_forecast/
├── __init__.py
├── configs/
│   └── default.yaml          # Master YAML config
├── data/
│   ├── __init__.py
│   ├── ingest.py             # goes2go download, reproject, normalize
│   └── dataset.py            # PyTorch sliding-window dataset
├── models/
│   ├── __init__.py
│   ├── blocks.py             # ResidualBlock, Attention, SSIM, PhysicsLoss
│   ├── encoders/
│   │   ├── __init__.py
│   │   └── spatial_cnn.py    # Land / Sea / Cloud CNN encoders
│   ├── temporal/
│   │   ├── __init__.py
│   │   └── probabilistic.py  # VRNN, BayesianLSTM, Transformer
│   ├── decoder/
│   │   ├── __init__.py
│   │   └── reverse_generator.py  # Conditional UNet decoder
│   └── fusion/
│       ├── __init__.py
│       └── fusion_model.py   # CrossAttention / ConcatMLP / FiLM
├── training/
│   ├── __init__.py
│   └── trainer.py            # Multi-stage training engine
├── evaluation/
│   ├── __init__.py
│   └── metrics.py            # RMSE, MAE, SSIM, CRPS, calibration
├── utils/
│   ├── __init__.py
│   ├── config.py             # YAML load / merge / validate
│   └── reproducibility.py    # Seeding, device, AMP, early stopping
├── scripts/
│   ├── __init__.py
│   ├── train.py              # Main training entry point
│   └── evaluate.py           # Evaluation entry point
├── tests/
│   └── __init__.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Dry-Run (Synthetic Data)

Smoke-test the full pipeline without downloading satellite data:

```bash
python -m goes_forecast.scripts.train --config configs/default.yaml --dry-run
```

### Full Training

```bash
# Train all four stages sequentially
python -m goes_forecast.scripts.train --config configs/default.yaml

# Resume from stage 2
python -m goes_forecast.scripts.train --config configs/default.yaml --stage 2
```

### Evaluation

```bash
python -m goes_forecast.scripts.evaluate --config configs/default.yaml --dry-run
```

---

## Configuration

All hyperparameters, data paths, model architectures, and training schedules are controlled through `configs/default.yaml`. Key sections:

- **data** — satellite, products, date range, grid, normalization, storage
- **encoder** — per-domain CNN architecture (land/sea/cloud)
- **temporal** — model type (variational_rnn / bayesian_lstm / transformer), dimensions, KL weight
- **decoder** — UNet depth, skip connections, activation
- **fusion** — fusion strategy (cross_attention / concat_mlp / film)
- **training** — per-stage epochs, LR, scheduler, loss weights
- **evaluation** — metrics list, calibration bins, horizons

---

## Evaluation Metrics

| Metric               | Type           | Description                              |
|----------------------|----------------|------------------------------------------|
| RMSE                 | Deterministic  | Root mean squared error                  |
| MAE                  | Deterministic  | Mean absolute error                      |
| SSIM                 | Structural     | Structural similarity index              |
| CRPS                 | Probabilistic  | Continuous Ranked Probability Score      |
| Calibration          | Probabilistic  | PIT histogram / reliability diagram      |
| Spatial Correlation  | Spatial        | Pearson correlation over spatial fields   |
| Multi-step degrad.   | Temporal       | Metric decay across forecast horizons    |

---

## Design Principles

1. **Physical plausibility** — physics-aware loss regularises spatial smoothness and temporal conservation; surface conditions, wind, and humidity are treated as core weather drivers.
2. **Uncertainty quantification** — the temporal model produces full distributions, not just point estimates; CRPS and calibration are first-class evaluation targets.
3. **Modularity** — each stage can be trained, checkpointed, and swapped independently; new data sources or model variants slot in through config and factory functions.
4. **Reproducibility** — global seeding, deterministic CUDNN, config snapshots saved with each run.
5. **Scalability** — mixed precision, configurable batch sizes, multi-worker data loading, Zarr storage for large time-series.

---

## License

MIT