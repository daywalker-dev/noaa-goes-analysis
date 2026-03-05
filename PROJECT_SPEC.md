# GOES-L2 Probabilistic Earth-System Forecasting Framework
## Project Specification, Architecture & Task Roadmap

| Field | Value |
|---|---|
| Version | 0.1.0 — Initial Specification |
| Status | Pre-Implementation |
| Data Source | GOES-16/17/18 via goes2go |
| Framework | PyTorch 2.x + Zarr + xarray |
| Training Stages | 4 |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Training Strategy](#5-training-strategy)
6. [Evaluation Framework](#6-evaluation-framework)
7. [Task Roadmap](#7-task-roadmap)
8. [Task Summary](#8-task-summary)
9. [Recommended Development Sequence](#9-recommended-development-sequence)

---

## 1. Project Overview

This framework is a modular, research-grade ML pipeline for probabilistic atmospheric forecasting using GOES-16/17/18 Level 2 satellite data. It learns surface and sea dynamics from multi-channel L2 products and produces uncertainty-aware multi-step forecasts with synthetic satellite-field reconstruction.

> **Goal:** The system functions as a physics-aware Earth-system emulator — `surface + wind + humidity → future atmospheric state → reconstructed GOES L2 fields`.

### Core Capabilities

- Ingests GOES L2 products (SST, LST, CMIP, DMW, TPW, AOD) via goes2go
- Learns spatial features per domain (land, sea, cloud) using CNN encoders
- Produces probabilistic multi-step atmospheric forecasts with uncertainty (μ ± σ)
- Reconstructs synthetic GOES L2 satellite fields from predicted atmospheric states
- Fuses all prediction streams into a unified holistic atmospheric forecast
- Evaluates with RMSE, MAE, SSIM, CRPS, calibration curves, and spatial correlation

### Design Principles

**Physical plausibility first.** Models are constrained by physics-aware losses and physically motivated architecture decisions.

**Modularity.** Each stage is independently trainable and swappable. New data sources can be added without touching core models.

**Uncertainty throughout.** Every forecast comes with calibrated uncertainty estimates. The system is designed for operational reliability.

**Config-driven.** All hyperparameters, data paths, model dims, and loss weights live in YAML configs — no magic numbers in code.

**Reproducibility.** Deterministic seeding, tracked checkpoints, and versioned configs ensure experiments are reproducible.

---

## 2. System Architecture

The system is composed of four sequentially trained components that together form an end-to-end probabilistic forecasting pipeline. Each component can be operated independently after training.

```
┌──────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                              │
│   goes2go → L2 Products → Align / Reproject → Normalize → Zarr     │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│             STAGE 1: SPATIAL ENCODERS                                │
│        land CNN  |  sea CNN  |  cloud/moisture CNN                   │
│  Multi-channel L2 → ResBlocks + CBAM → Latent z + reconstruction    │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ latent embeddings
┌────────────────────────────────▼─────────────────────────────────────┐
│             STAGE 2: VARIATIONAL TRANSFORMER                         │
│  Latents + Wind + Humidity + Temp → ELBO → μ(state) + σ²(state)    │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ predicted atmospheric state
┌────────────────────────────────▼─────────────────────────────────────┐
│             STAGE 3: CONDITIONAL UNET (reverse generator)            │
│       Predicted state → FiLM + UNet → Synthetic GOES L2 fields      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│             STAGE 4: FUSION TRANSFORMER                              │
│  Original L2 + CNN preds + Bayesian outputs + Wind/humidity          │
│              → Unified forecast + uncertainty maps                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1 — Spatial CNN Encoders

Three independent encoder instances trained on separate GOES L2 domains. Each encoder accepts multi-channel L2 patches, learns a compressed latent representation, and simultaneously predicts the next timestep's fields.

| Property | Value |
|---|---|
| Architecture | Conv stem → DownBlocks with ResidualBlocks → CBAM attention → Global avg pool → Latent FC |
| Domains | Land (LST-based), Sea (SST-based), Cloud/Moisture (CMIP bands 7–16) |
| Outputs | Latent embedding `[B, latent_dim]` + next-step reconstruction `[B, C, H, W]` |
| Attention | CBAM = Channel Attention (squeeze-excitation) + Spatial Attention (max+avg pooling) |
| Loss | `α·MSE + β·SSIM + γ·PhysicsConstraint` |
| Frozen after | Stage 1 completes; encoders frozen for Stages 2 and 3 |

---

### Stage 2 — Variational Transformer

The core probabilistic forecasting model. Takes a time-series of spatial latents (from frozen encoders) plus meteorological fields, and outputs a full distribution over future atmospheric states for each forecast step.

| Property | Value |
|---|---|
| Architecture | Input projection → Sinusoidal PE → Transformer encoder → VAE bottleneck → Transformer decoder → Output heads |
| Inputs | Spatial latents `[B, T_in, D]` + meteo fields `[B, T_in, F]` (wind U/V, humidity, temp, pressure) |
| VAE bottleneck | μ_z and log σ²_z via linear heads on pooled memory; reparameterization trick during training |
| Outputs | μ `[B, T_out, state_dim]` + log σ² `[B, T_out, state_dim]` per forecast step |
| Uncertainty | Monte Carlo sampling at inference (N=50 samples) for mean, std, and percentile bands |
| Loss | `ELBO = Reconstruction loss − β·KL(q‖p)` with free-bits regularization |
| Autoregression | Learnable forecast query tokens drive the decoder |

---

### Stage 3 — Conditional UNet (Reverse Generator)

A conditional image-generation network that maps predicted atmospheric state vectors back into the spatial domain, reconstructing synthetic GOES L2 satellite fields. The atmospheric state is injected via FiLM at the bottleneck.

| Property | Value |
|---|---|
| Architecture | Learnable spatial prior → FiLM conditioning → Upsampling stages with ResBlocks → Output head |
| Conditioning | FiLM: per-channel scale γ and shift β derived from state vector |
| Inputs | Predicted state vector `[B, state_dim]` + optional noise `[B, noise_dim]` |
| Outputs | Synthetic L2 field stack `[B, out_channels, H, W]` |
| Stability | Optional spectral normalization on output head |
| Loss | `MSE + SSIM + SpectralLoss` (2D FFT magnitude matching) |

---

### Stage 4 — Fusion Transformer

The final integration layer. Fuses four distinct prediction streams into a single coherent atmospheric forecast. Each source is projected to a common dimension, enriched with source-type embeddings, and fused via a lightweight transformer.

| Property | Value |
|---|---|
| Sources | 1) CNN latents, 2) Bayesian μ + σ², 3) Meteorological fields, 4) Generator reconstructions |
| Architecture | Source projectors → Source embeddings → Transformer encoder → Forecast + uncertainty heads |
| Fusion strategy | All `T_out × N_sources` tokens processed jointly; mean-aggregated to `[B, T_out, D]` |
| Forecast head | Per-channel spatial scales applied to generator output as spatial prior |
| Uncertainty head | Predictive uncertainty modulated by Bayesian variance |
| Loss | `MSE + SSIM + CRPS + PhysicsConstraint` |

---

## 3. Project Structure

```
noaa-goes-analysis/
├── README.md                        Architecture overview and quickstart
├── requirements.txt                 Python dependencies
├── setup.py                         Package installation config
├── main.py                         Main python file
│
├── config/
│   ├── base_config.yaml             Master config: data, model, training, eval
│   ├── encoder_config.yaml          Spatial encoder architecture overrides
│   ├── temporal_config.yaml         Variational transformer hyperparameters
│   └── experiment_config.yaml       Per-experiment override configs
│
├── data/
│   ├── downloader.py                goes2go wrapper — fetches GOES L2 products
│   ├── preprocessor.py              Reproject, align, normalize, write Zarr
│   ├── dataset.py                   GOESWindowDataset: sliding windows
│   ├── zarr_store.py                Zarr read/write and chunking utilities
│   └── augmentation.py              Spatial/temporal augmentation transforms
│
├── models/
│   ├── blocks.py                    ResidualBlock, CBAM, FiLM, SinusoidalPE
│   ├── spatial_encoder.py           SpatialCNNEncoder + DomainEncoderEnsemble
│   ├── temporal_bayesian.py         VariationalTransformer with ELBO
│   ├── reverse_generator.py         ConditionalUNet for L2 reconstruction
│   └── fusion.py                    FusionTransformer for holistic forecast
│
├── training/
│   ├── losses.py                    MSE, SSIM, CRPS, Spectral, Physics, ELBO
│   ├── trainer.py                   BaseTrainer with AMP, clipping, checkpointing
│   ├── stage_runners.py             EncoderRunner, TemporalRunner, GeneratorRunner, FusionRunner
│   └── callbacks.py                 LR warmup, early stopping, logging hooks
│
├── evaluation/
│   ├── metrics.py                   RMSE, MAE, SSIM, CRPS, spatial correlation
│   ├── calibration.py               Reliability diagrams, rank histograms
│   └── visualizer.py                Forecast maps, uncertainty bands, skill curves
│
├── utils/
│   ├── config_loader.py             YAML parsing with OmegaConf + CLI overrides
│   ├── logger.py                    Structured logging with rich formatting
│   ├── reproducibility.py           Seed control, deterministic mode
│   └── projection.py                GOES geostationary ↔ lat/lon reprojection
│
└── scripts/
    ├── download_data.py             Fetch GOES L2 products for a date range
    ├── preprocess.py                Run preprocessing pipeline → Zarr store
    ├── train.py                     Orchestrate all training stages
    └── evaluate.py                  Run full evaluation suite on a checkpoint
```

---

## 4. Data Pipeline

### GOES L2 Products

| Product ID | Description | Channels | Used For |
|---|---|---|---|
| `ABI-L2-SSTF` | Sea Surface Temperature | SST, DQF | Sea encoder |
| `ABI-L2-LSTF` | Land Surface Temperature | LST, DQF | Land encoder |
| `ABI-L2-CMIPF` | Cloud/Moisture Imagery | CMI (bands 7–16), DQF | Cloud encoder |
| `ABI-L2-DMWF` | Derived Motion Winds | wind_speed, wind_direction, pressure, DQF | Temporal meteo |
| `ABI-L2-TPWF` | Total Precipitable Water | TPW, DQF | Temporal meteo |
| `ABI-L2-AODF` | Aerosol Optical Depth | AOD, DQF | Optional supplement |

### Preprocessing Steps

1. Download raw NetCDF files using goes2go for each product across the training date range
2. Parse DQF (Data Quality Flag); discard scenes with valid pixel fraction below threshold
3. Reproject from GOES geostationary fixed-grid to a common lat/lon grid using pyresample
4. Temporally align all products to a 1-hourly axis; forward-fill gaps up to 30 minutes
5. Compute per-channel z-score statistics over training split only; persist with Zarr store
6. Apply normalization; append quality mask as an additional channel per variable
7. Write to chunked Zarr store with dimensions `(time, channel, lat, lon)`
8. Validate: check completeness, run spot-check visualizations, confirm channel ordering

### Dataset Configuration

| Parameter | Value |
|---|---|
| `input_steps` | 12 hourly steps of context (12h lookback) |
| `forecast_steps` | 24 hourly steps of prediction (24h horizon) |
| `patch_size` | 256 × 256 spatial patches; stride 128 (50% overlap) |
| Normalization | z-score per channel; stats from training split only |
| Missing data | Configurable: interpolate / zero-fill / mask |
| Augmentation | Horizontal flip, vertical flip (training only) |

---

## 5. Training Strategy

Training proceeds in four sequential stages. Each stage builds on frozen outputs from the previous. This prevents gradient conflicts, accelerates convergence, and allows independent debugging.

| Stage | Component | Epochs | Frozen Deps | Key Loss |
|---|---|---|---|---|
| 1 | Spatial CNN Encoders | 50 | None | MSE + SSIM + Physics |
| 2 | Variational Transformer | 80 | Encoders (frozen) | ELBO + CRPS |
| 3 | Conditional UNet Generator | 50 | Encoders + Temporal (frozen) | MSE + SSIM + Spectral |
| 4 | Fusion Transformer (full) | 30 | Optionally unfreeze all | MSE + SSIM + CRPS + Physics |

### Training Infrastructure

- Mixed precision (bfloat16) via PyTorch AMP
- Gradient clipping (max norm = 1.0)
- Top-k checkpoint management with monitored metric (default: `val/crps`)
- WandB and TensorBoard logging, toggled via config
- All hyperparameters loaded from YAML; CLI overrides via `--override key=value`
- Global seed set for `random`, `numpy`, `torch`, and CUDA

### Loss Function Reference

| Loss | Description |
|---|---|
| `MaskedMSE` | MSE applied only to valid (non-masked) pixels |
| `SSIMLoss` | `1 − SSIM`; uses pytorch-msssim with 3×3 pooling fallback |
| `CRPSLoss (Gaussian)` | Closed-form CRPS for Gaussian distributions |
| `SpectralLoss` | L2 distance between 2D FFT magnitude spectra |
| `PhysicsConstraint` | Temporal smoothness + energy conservation proxy + spatial gradient consistency |
| `ELBOLoss` | Reconstruction + `β·KL(q‖p)` with free-bits regularization |
| `CompositeLoss` | Weighted sum builder; weights loaded from YAML per stage |

---

## 6. Evaluation Framework

| Metric | Type | Measures | Ideal |
|---|---|---|---|
| RMSE | Deterministic | Magnitude of forecast error | 0 |
| MAE | Deterministic | Robust error; less sensitive to outliers | 0 |
| Bias | Deterministic | Systematic over/under-prediction | 0 |
| SSIM | Spatial quality | Structural similarity of forecast vs observed | 1 |
| Spatial ACC | Spatial quality | Anomaly correlation coefficient | 1 |
| CRPS | Probabilistic | Joint accuracy + sharpness of predictive distribution | 0 |
| Coverage (90%) | Calibration | Fraction of obs in 90% CI — should be ≈ 0.9 | 0.9 |
| Reliability | Calibration | Reliability diagram slope — 1:1 = perfect | 1:1 |
| Step degradation | Multi-step | How RMSE/CRPS grow with lead time | Flat |

### Evaluation Outputs

- Per-variable skill scores at lead times 1h, 3h, 6h, 12h, 24h
- Spatial maps of mean absolute error and uncertainty coverage
- Reliability diagrams per forecast step and per L2 channel
- Rank histograms for ensemble consistency verification
- Multi-step degradation curves (RMSE and CRPS vs lead time)
- Calibration summary: nominal vs actual coverage at 50%, 80%, 90%, 95% CI
- Comparison to persistence baseline

---

## 7. Task Roadmap

Tasks are organized by stage and annotated with priority, effort, and subtask checklists.

### Priority Key

| Label | Meaning |
|---|---|
| 🔴 Critical | Blocking — nothing proceeds without it |
| 🟠 High | Required for a working system; do early |
| 🔵 Medium | Improves quality/robustness; schedule after High |
| 🟢 Low | Nice to have; defer to later iterations |

### Effort Key

`XS` < 2h · `S` half-day · `M` 1–2 days · `L` 3–5 days · `XL` 1–2 weeks

---

### Stage 0 — Project Setup & Environment

---

#### T-001 · Environment and Dependency Setup · 🔴 Critical · `S`

Create the conda/venv environment, install all requirements, verify goes2go connectivity, and confirm PyTorch detects available GPU hardware.

- [ ] Create virtual environment (Python 3.11+) or conda env
- [ ] Install core dependencies: `torch`, `goes2go`, `xarray`, `zarr`, `numcodecs`, `pyresample`, `pyproj`
- [ ] Install optional dependencies: `pytorch-msssim`, `properscoring`, `wandb`, `cmocean`
- [ ] Run goes2go quickstart test to confirm AWS S3 bucket access
- [ ] Verify CUDA availability and bfloat16 support
- [ ] Set up `.env` file for data and output paths

---

#### T-002 · Repository Structure and Package Configuration · 🔴 Critical · `S`

Initialize the repository with the full directory structure, create all `__init__.py` files, configure `setup.py`, and add `.gitignore` for large data files and checkpoints.

- [ ] Create all module directories and `__init__.py` files
- [ ] Write `setup.py` or `pyproject.toml` with package metadata
- [ ] Configure `.gitignore` to exclude `data/`, `outputs/`, `*.ckpt`, `*.zarr`, `*.nc`
- [ ] Create `config/` with `base_config.yaml` skeleton
- [ ] Initialize git repository with initial commit

---

#### T-003 · Configuration System · 🟠 High · `M`

Implement the YAML configuration loader using OmegaConf with support for CLI overrides, config merging, and validation via Pydantic schemas.

- [ ] Implement `utils/config_loader.py` with OmegaConf loading
- [ ] Support `--override key=value` syntax for CLI experiments
- [ ] Add Pydantic model schemas for `DataConfig`, `ModelConfig`, `TrainingConfig`
- [ ] Validate config on load; raise helpful errors for missing required fields
- [ ] Write unit tests for config loading and override merging

---

#### T-004 · Logging and Reproducibility Utilities · 🟠 High · `S`

Set up structured logging with rich formatting, implement global seed setting, and create a utility to capture and log the full config at the start of each run.

- [ ] Implement `utils/logger.py` with rich handler for terminal output
- [ ] Implement `utils/reproducibility.py`: set seeds for `random`, `numpy`, `torch`, CUDA
- [ ] Log full resolved config to `output_dir/config.yaml` at run start
- [ ] Add experiment ID generation (timestamp + short hash) for run isolation

---

### Stage 1 — Data Pipeline

---

#### T-010 · GOES L2 Downloader · 🔴 Critical · `M`

Implement `GOESDownloader` in `data/downloader.py` to fetch all configured L2 products for arbitrary date ranges, with retry logic, skip-existing support, and a file manifest output.

- [ ] Implement `GOESDownloader.download()` using `goes2go GOES.timerange()`
- [ ] Handle optional products (AOD) gracefully when unavailable
- [ ] Add retry with exponential backoff for transient network failures
- [ ] Implement `build_manifest()` returning a DataFrame of `(product, path, timestamp, size_mb)`
- [ ] Write CLI `scripts/download_data.py` with `--start`, `--end`, `--products`, `--output-dir` flags
- [ ] Add progress bars using tqdm for multi-product downloads
- [ ] Write integration test downloading 1 day of SST data

---

#### T-011 · Spatial Reprojection to Common Grid · 🔴 Critical · `L`

Implement accurate reprojection from GOES geostationary fixed-grid to the target lat/lon grid using pyresample for production quality; scipy bilinear for testing.

- [ ] Parse GOES projection metadata from `goes_imager_projection` variable
- [ ] Implement pyresample-based area definition for the target lat/lon grid
- [ ] Implement bilinear interpolation fallback using `scipy.interpolate`
- [ ] Handle NaN propagation through reprojection (mask-aware interpolation)
- [ ] Validate reprojection against known geographic features (coastlines)
- [ ] Benchmark reprojection speed per file; optimize if > 5s per scene

---

#### T-012 · Quality Flag Processing and Missing Data · 🟠 High · `M`

Parse DQF channels from each product, compute per-scene quality fractions, and implement the three fill strategies (interpolate, zero-fill, mask).

- [ ] Extract DQF from each L2 product; validate `DQF == 0` means good quality
- [ ] Compute `valid_fraction`; filter scenes below threshold
- [ ] Implement nearest-neighbour fill (`distance_transform_edt`) for interpolate strategy
- [ ] Append binary mask channel per variable (1 = valid, 0 = filled/missing)
- [ ] Validate filled scenes look physically reasonable with spot-check plots

---

#### T-013 · Temporal Alignment and Zarr Store Writer · 🔴 Critical · `M`

Align all products to a common 1-hourly time axis and write the merged, normalized result to a chunked Zarr store.

- [ ] Round all file timestamps to the nearest hour for alignment
- [ ] Reindex each product DataArray to the common time axis with 30-min tolerance
- [ ] Concatenate along channel dimension in configured order
- [ ] Compute and persist z-score statistics (JSON file alongside Zarr store)
- [ ] Write Zarr with Blosc LZ4 compression; validate chunk sizes for sequential access
- [ ] Implement `zarr_store.py` with read utilities (`open_zarr`, `get_stats`, `list_channels`)
- [ ] Write CLI `scripts/preprocess.py` accepting manifest path and output path

---

#### T-014 · GOESWindowDataset and DataLoader · 🔴 Critical · `M`

Implement the sliding window PyTorch Dataset that serves `(input_window, target_window)` pairs from the Zarr store, with spatial patching and augmentation.

- [ ] Implement `_build_window_indices()` for all `(time, lat, lon)` patch combinations
- [ ] Implement `__getitem__` with lazy Zarr reads, NaN replacement, and mask extraction
- [ ] Test patch extraction at boundaries; ensure no out-of-bounds indexing
- [ ] Implement random flip augmentation for training split
- [ ] Implement `build_dataloader()` factory for train/val/test splits
- [ ] Profile DataLoader throughput; ensure GPU is not bottlenecked by data loading
- [ ] Write unit tests for index construction, item shape, and split date filtering

---

#### T-015 · Data Validation and Visualization · 🔵 Medium · `S`

Build diagnostic scripts to validate the Zarr store contents and visualize sample scenes before investing GPU compute in training.

- [ ] Script to plot a sample input/target window for each L2 product
- [ ] Script to compute and display per-channel statistics from the Zarr store
- [ ] Script to visualize the geographic coverage of reprojected fields
- [ ] Check temporal coverage: histogram of valid scenes per month/day
- [ ] Verify mask channels correlate with visually cloudy/missing regions

---

### Stage 2 — Spatial CNN Encoder Models

---

#### T-020 · Shared Building Blocks · 🔴 Critical · `M`

Implement all reusable model components in `models/blocks.py`. These are depended upon by all four model stages.

- [ ] Implement `ResidualBlock` with pre-activation (GroupNorm → GELU → Conv → Conv)
- [ ] Implement `ChannelAttention` (squeeze-excitation with avg + max pooling)
- [ ] Implement `SpatialAttention` (channel-pooled conv gate)
- [ ] Combine into `CBAM` module; verify gradient flow through attention gates
- [ ] Implement `FiLM` conditioning layer with γ (scale) and β (shift) projections
- [ ] Implement `SinusoidalPE` for transformer positional encoding
- [ ] Implement `DownBlock` and `UpBlock`
- [ ] Write shape-in/shape-out unit tests for every block

---

#### T-021 · SpatialCNNEncoder Architecture · 🔴 Critical · `M`

Implement `SpatialCNNEncoder` with the full encoder-decoder architecture: Conv stem → downsampling with CBAM → global avg pool → latent → decoder with skip connections.

- [ ] Implement encoder stem (7×7 conv, GroupNorm, GELU)
- [ ] Implement N encoder stages (DownBlock + `n_res_blocks` ResBlocks + CBAM)
- [ ] Implement latent projection: global avg pool → LayerNorm → 2-layer FC → latent z
- [ ] Implement decoder: `latent_to_spatial` projection → UpBlocks with skip connections
- [ ] Handle spatial size mismatches between encoder/decoder via bilinear resize
- [ ] Implement `encode_only()` method for frozen encoder inference
- [ ] Test: input `[4, 3, 256, 256]` → latent `[4, 128]`, recon `[4, 3, 256, 256]`

---

#### T-022 · DomainEncoderEnsemble · 🔴 Critical · `S`

Wrap three `SpatialCNNEncoder` instances (land, sea, cloud) into a `DomainEncoderEnsemble` that routes domain-specific inputs and concatenates latents.

- [ ] Implement `DomainEncoderEnsemble` with `nn.ModuleDict` of domain encoders
- [ ] Implement `forward()` accepting dict of domain inputs; return latents, recons, combined
- [ ] Implement `freeze()` and `unfreeze()` methods for staged training
- [ ] Validate combined latent dimension = `n_domains × latent_dim`

---

#### T-023 · Encoder Training Stage Runner · 🔴 Critical · `L`

Implement `EncoderStageRunner` subclassing `BaseTrainer`, with domain channel splitting, composite loss computation, and optimizer/scheduler setup from config.

- [ ] Implement `_train_step`: split batch by domain, run encoders, compute losses
- [ ] Implement `CompositeLoss` with configurable MSE + SSIM + PhysicsConstraint weights
- [ ] Implement `PhysicsConstraintLoss`: temporal smoothness + energy proxy + gradient consistency
- [ ] Integrate `SSIMLoss` with pytorch-msssim; fall back gracefully if not installed
- [ ] Run first training loop end-to-end; verify loss decreases over 10 steps
- [ ] Profile memory usage; ensure `batch_size=4` fits on single GPU with AMP

---

#### T-024 · Encoder Training Run and Validation · 🟠 High · `L`

Execute Stage 1 training to convergence and validate that latent representations are meaningful and reconstructions are visually coherent.

- [ ] Train for full 50 epochs on training split; monitor val loss for convergence
- [ ] Visualize reconstructions vs originals at epochs 1, 10, 25, 50
- [ ] Compute RMSE and SSIM of reconstructions on validation set
- [ ] Visualize t-SNE of latent vectors colored by season / time-of-day / region
- [ ] Confirm latents encode physically meaningful variation (not random noise)
- [ ] Save final checkpoint; freeze encoders for Stage 2

---

### Stage 3 — Variational Transformer (Temporal Model)

---

#### T-030 · Variational Transformer Architecture · 🔴 Critical · `L`

Implement `VariationalTransformer` in `models/temporal_bayesian.py` with the full VAE-in-Transformer architecture: input projection, PE, transformer encoder, VAE bottleneck, transformer decoder, output heads.

- [ ] Implement `MeteoProjector`: MLP projecting F meteorological fields to `meteo_dim`
- [ ] Implement input fusion: latent projection + meteo projection (additive combination)
- [ ] Configure `TransformerEncoderLayer` with pre-norm (`norm_first=True`) for stability
- [ ] Implement VAE bottleneck: `to_mu` and `to_logvar` on mean-pooled memory
- [ ] Implement reparameterization trick (train) vs MAP estimate (eval)
- [ ] Implement learnable forecast query tokens for autoregressive decoding
- [ ] Implement output heads: `out_mu` and `out_logvar` with log-variance clamping
- [ ] Test: input `[4, 12, 384]` + `[4, 12, 5]` → mu `[4, 24, 256]` + logvar `[4, 24, 256]`

---

#### T-031 · ELBO and CRPS Loss Functions · 🔴 Critical · `M`

Implement `ELBOLoss` with free-bits KL regularization and closed-form `CRPSLoss` for Gaussian distributions.

- [ ] Implement `ELBOLoss.forward()` returning `dict{total, recon, kl}`
- [ ] Add free-bits: clamp per-dim KL from below at `free_bits` nats (default 0.5)
- [ ] Implement β-VAE weighting; support KL annealing schedule (linear warm-up to β over N epochs)
- [ ] Implement `CRPSLoss` using `scipy.stats Normal` CDF/PDF for closed-form computation
- [ ] Write numerical tests: CRPS of perfect forecast = 0; deteriorates with wrong σ
- [ ] Validate ELBO decreases during first 100 training steps

---

#### T-032 · Monte Carlo Inference and Uncertainty · 🟠 High · `M`

Implement `sample()` method for Monte Carlo uncertainty estimation at inference time.

- [ ] Implement `sample()` with `train()` mode for stochastic VAE sampling
- [ ] Compute per-step mean, std, p05, p95 across N samples
- [ ] Benchmark sampling speed: N=50 samples on a single batch in < 2s
- [ ] Validate that uncertainty grows with forecast lead time
- [ ] Validate that uncertainty is smaller for constrained fields (SST) vs noisy fields (winds)

---

#### T-033 · Temporal Model Training Stage Runner · 🔴 Critical · `L`

Implement `TemporalStageRunner`, including frozen encoder inference, state target construction, and combined ELBO + CRPS training.

- [ ] Implement `_extract_latents()`: loop input window, run frozen encoders, stack to `[B, T, D]`
- [ ] Implement `_extract_meteo()`: spatial avg-pool meteorological channels to `[B, T, F]`
- [ ] Construct state target as spatial-avg-pooled L2 channels `[B, T_out, C]`
- [ ] Align μ/logvar output dims to match target state dimensionality
- [ ] Implement KL annealing: linear warm-up from 0 to β over first 20 epochs
- [ ] Train for 80 epochs; monitor recon loss, KL, and CRPS separately in logs

---

#### T-034 · Temporal Model Validation · 🟠 High · `M`

Validate probabilistic forecast quality on the validation set using CRPS, coverage scores, and reliability diagrams.

- [ ] Compute CRPS on validation set for all variables at each forecast step
- [ ] Plot CRPS vs lead time; confirm gradual degradation (not flat or explosive)
- [ ] Compute 90% CI coverage: should be close to 0.90 for calibrated forecasts
- [ ] Plot reliability diagrams per channel; identify over- or under-dispersed channels
- [ ] Compare against persistence baseline CRPS; confirm model adds skill

---

### Stage 4 — Reverse Generator (UNet)

---

#### T-040 · Conditional UNet Architecture · 🔴 Critical · `L`

Implement `ConditionalUNet` in `models/reverse_generator.py` with a learnable spatial prior, FiLM conditioning at each decoder level, and a spectral-normalized output head.

- [ ] Implement learnable `spatial_prior` parameter of shape `[1, C_bottle, h0, w0]`
- [ ] Implement FiLM injection at each upsampling level: project `cond_emb` → γ, β
- [ ] Implement condition embedding: `concat(state, noise)` → MLP → LayerNorm → `cond_emb`
- [ ] Implement N upsampling stages (bilinear up → Conv → GroupNorm → GELU → ResBlock)
- [ ] Implement output head with optional spectral normalization (`nn.utils.spectral_norm`)
- [ ] Implement `decode_sequence()` for decoding an entire `[B, T, D]` state sequence
- [ ] Test: condition `[4, 256]` → output `[4, 20, 256, 256]`

---

#### T-041 · Spectral Loss · 🔵 Medium · `S`

Implement `SpectralLoss` computing L2 distance between 2D FFT magnitude spectra.

- [ ] Implement `SpectralLoss` using `torch.fft.rfft2` with `norm='ortho'`
- [ ] Test on random noise vs structured image: spectral loss should be much lower for structured
- [ ] Tune spectral loss weight to avoid dominating SSIM; default 0.2

---

#### T-042 · Generator Training Stage Runner · 🔴 Critical · `M`

Implement `GeneratorStageRunner` with frozen encoder + temporal model inference, then generator MSE + SSIM + Spectral training.

- [ ] Implement `_get_predicted_states()` using frozen encoder + temporal model
- [ ] Reshape targets for per-step SSIM computation `[B*T, C, H, W]`
- [ ] Train for 50 epochs; monitor reconstruction SSIM on validation set
- [ ] Save worst-case reconstruction examples for failure analysis

---

#### T-043 · Generator Validation and Visual Quality · 🟠 High · `M`

Evaluate the generator's ability to reconstruct spatially coherent, physically plausible L2 fields.

- [ ] Side-by-side visual comparison of generated vs true L2 fields for SST, LST, TPW
- [ ] Compute RMSE, SSIM, spectral energy ratio per variable on validation set
- [ ] Check for mode collapse: generate 20 samples for same state; verify diversity
- [ ] Identify systematic biases (e.g. blurry coastlines); diagnose and address
- [ ] Profile memory: batch decoding 24 steps should fit in GPU memory

---

### Stage 5 — Fusion Model

---

#### T-050 · Fusion Transformer Architecture · 🔴 Critical · `L`

Implement `FusionTransformer` in `models/fusion.py` with four source projectors, source-type embeddings, transformer-based fusion, and dual forecast + uncertainty output heads.

- [ ] Implement four source projectors: `cnn_proj`, `bayes_proj`, `meteo_proj`, `gen_proj`
- [ ] Implement `source_embeddings`: `nn.Embedding(n_sources=4, d_model)`
- [ ] Reshape inputs to `[B, T * n_sources, D]` for joint transformer processing
- [ ] Mean-aggregate across source dimension back to `[B, T, D]`
- [ ] Implement `forecast_head`: linear → spatial scale applied to generator output
- [ ] Implement `uncertainty_head`: softplus output modulated by Bayesian variance
- [ ] Test: all inputs `[B=2, T=24, …]` → forecast `[2, 24, 20, 256, 256]`

---

#### T-051 · Fusion Training Stage Runner · 🔴 Critical · `L`

Implement `FusionStageRunner` orchestrating all sub-models, with optional full unfreezing for end-to-end fine-tuning.

- [ ] Implement `_forward_all()`: run encoder → temporal → generator → fusion in sequence
- [ ] Handle temporal dimension expansion: CNN latents for forecast steps via last-step repeat
- [ ] Implement combined MSE + SSIM + CRPS + Physics loss with config weights
- [ ] Support `unfreeze_all` flag to allow fine-tuning all sub-models jointly
- [ ] Monitor GPU memory during fusion forward pass; add gradient checkpointing if needed

---

#### T-052 · Full System Integration Test · 🔴 Critical · `M`

Run a complete forward pass through all four stages end-to-end with real data.

- [ ] Load all four checkpoints; run end-to-end inference on 1 batch
- [ ] Verify forecast and uncertainty output shapes match expected `[B, T, C, H, W]`
- [ ] Confirm gradients flow through all parameters during fusion backward pass
- [ ] Run gradient norm checks: no exploding or vanishing gradients
- [ ] Log memory footprint at each stage; document minimum GPU VRAM requirement

---

### Stage 6 — Evaluation and CLI Scripts

---

#### T-060 · Metrics Module · 🟠 High · `M`

Implement all evaluation metrics in `evaluation/metrics.py`.

- [ ] Implement RMSE, MAE, bias with optional mask argument
- [ ] Implement `ssim_score` using `scipy.ndimage.uniform_filter` approximation
- [ ] Implement `spatial_correlation` (anomaly correlation coefficient)
- [ ] Implement `crps_gaussian` using `scipy.stats.norm`
- [ ] Implement `crps_ensemble` using energy form for Monte Carlo samples
- [ ] Implement `coverage_score` for arbitrary confidence levels
- [ ] Implement `multistep_skill`: compute metric at each lead time
- [ ] Write unit tests with known analytic results for CRPS and SSIM

---

#### T-061 · Calibration Analysis · 🟠 High · `M`

Implement `reliability_diagram` and `rank_histogram` in `evaluation/calibration.py`.

- [ ] Implement `reliability_diagram`: bin forecasts by quantile; measure actual hit rates
- [ ] Implement `rank_histogram` (Talagrand diagram) for ensemble spread assessment
- [ ] Compute integrated calibration error (ICE) as scalar summary
- [ ] Write sharpness metric: mean width of 80% CI across all forecast times

---

#### T-062 · Visualization Module · 🔵 Medium · `L`

Build `evaluation/visualizer.py` with spatial forecast maps, uncertainty bands, skill curves, and calibration diagrams.

- [ ] Implement `plot_forecast_map`: side-by-side forecast vs truth with error overlay
- [ ] Implement `plot_uncertainty_bands`: time series with shaded 50/80/90% CI
- [ ] Implement `plot_skill_curves`: RMSE and CRPS vs lead time per variable
- [ ] Implement `plot_reliability_diagram`: nominal vs observed coverage with 1:1 reference
- [ ] Implement `plot_spatial_error`: geographic heatmap of mean absolute error
- [ ] Use cmocean perceptually uniform colormaps for all meteorological fields

---

#### T-063 · CLI Entry Points · 🟠 High · `M`

Implement the four CLI scripts using Click.

- [ ] `scripts/download_data.py`: `--satellite`, `--start`, `--end`, `--products`, `--output-dir`, `--n-threads`
- [ ] `scripts/preprocess.py`: `--input`, `--output`, `--config`; run full pipeline
- [ ] `scripts/train.py`: `--stage` (encoders|temporal|generator|fusion|all), `--config`, `--override`, `--resume`
- [ ] `scripts/evaluate.py`: `--checkpoint`, `--split`, `--lead-times`, `--n-samples`, `--output-dir`
- [ ] Ensure all CLIs print helpful usage strings and validate inputs before running

---

#### T-064 · Persistence Baseline · 🔵 Medium · `S`

Implement a trivial persistence baseline (last observed L2 state repeated for all forecast steps).

- [ ] Implement `PersistenceBaseline` class returning last input frame for all `T_out` steps
- [ ] Evaluate baseline on test set with all metrics at all lead times
- [ ] Add baseline results to all skill curve plots as a dashed reference line

---

### Stage 7 — Testing, Documentation, and Hardening

---

#### T-070 · Unit and Integration Test Suite · 🟠 High · `L`

Write a comprehensive test suite using pytest covering all model shapes, data pipeline correctness, loss function values, and end-to-end integration.

- [ ] `tests/test_blocks.py`: shape tests for every building block
- [ ] `tests/test_encoders.py`: input → latent → recon shape; gradient flow test
- [ ] `tests/test_temporal.py`: forward pass shapes, KL > 0, CRPS > 0
- [ ] `tests/test_generator.py`: condition → output shape; `decode_sequence` shape
- [ ] `tests/test_fusion.py`: all-source forward pass; uncertainty output present
- [ ] `tests/test_losses.py`: CRPS of perfect forecast ≈ 0; SSIM(x, x) = 1
- [ ] `tests/test_dataset.py`: item shapes, split date filtering, augmentation
- [ ] `tests/test_preprocessor.py`: mock NetCDF → reprojected grid shape
- [ ] Configure pytest with pytest-cov; target > 80% code coverage

---

#### T-071 · README and Architecture Documentation · 🟠 High · `M`

Write comprehensive `README.md` covering project purpose, architecture, data flow, training commands, evaluation, and extension guide.

- [ ] Architecture section with ASCII data flow diagram
- [ ] Data requirements section: GOES products, date ranges, storage estimates
- [ ] Training section: all four stage commands with example configs
- [ ] Evaluation section: interpreting CRPS, calibration, and skill curves
- [ ] Extension guide: adding new data products, encoder domains, metrics
- [ ] Contribution guide: coding style (ruff + mypy), PR process, test requirements

---

#### T-072 · Type Hints and Static Analysis · 🔵 Medium · `M`

Add complete type hints to all public functions; configure mypy in strict mode and ruff for linting.

- [ ] Add return type annotations to all public functions
- [ ] Add parameter type annotations including Tensor shape hints in docstrings
- [ ] Configure `mypy.ini` in strict mode; resolve all type errors
- [ ] Configure ruff with rule set (E, F, I, UP, B); add pre-commit hook
- [ ] Run type checks in CI/CD pipeline

---

#### T-073 · Performance Profiling and Optimization · 🟢 Low · `L`

Profile training throughput for each stage and identify bottlenecks.

- [ ] Profile Stage 1 training: identify whether bottleneck is data loading or GPU compute
- [ ] Tune `num_workers` and `pin_memory` for DataLoader throughput
- [ ] Test `torch.compile()` on encoder and temporal model; measure speedup vs overhead
- [ ] Consider gradient checkpointing for Stage 4 fusion if GPU memory is tight
- [ ] Document minimum hardware requirements: GPU VRAM, CPU RAM, disk space

---

#### T-074 · Experiment Tracking and Model Registry · 🟢 Low · `M`

Integrate WandB for experiment tracking and add a simple model registry.

- [ ] Integrate `wandb.init` and `wandb.log` in `BaseTrainer`; gate behind config flag
- [ ] Log all hyperparameters, config YAML, and environment info at run start
- [ ] Log validation metric curves and sample visualizations every 5 epochs
- [ ] Implement a `model_registry.json` that indexes checkpoints by stage + metric
- [ ] Add `evaluate.py` flag to write evaluation results back to the registry

---

## 8. Task Summary

| ID | Title | Priority | Effort | Stage |
|---|---|---|---|---|
| T-001 | Environment and Dependency Setup | 🔴 Critical | S | 0 – Setup |
| T-002 | Repository Structure | 🔴 Critical | S | 0 – Setup |
| T-003 | Configuration System | 🟠 High | M | 0 – Setup |
| T-004 | Logging and Reproducibility | 🟠 High | S | 0 – Setup |
| T-010 | GOES L2 Downloader | 🔴 Critical | M | 1 – Data |
| T-011 | Spatial Reprojection | 🔴 Critical | L | 1 – Data |
| T-012 | Quality Flag Processing | 🟠 High | M | 1 – Data |
| T-013 | Temporal Alignment and Zarr Store | 🔴 Critical | M | 1 – Data |
| T-014 | GOESWindowDataset and DataLoader | 🔴 Critical | M | 1 – Data |
| T-015 | Data Validation and Visualization | 🔵 Medium | S | 1 – Data |
| T-020 | Shared Model Building Blocks | 🔴 Critical | M | 2 – Encoders |
| T-021 | SpatialCNNEncoder Architecture | 🔴 Critical | M | 2 – Encoders |
| T-022 | DomainEncoderEnsemble | 🔴 Critical | S | 2 – Encoders |
| T-023 | Encoder Training Stage Runner | 🔴 Critical | L | 2 – Encoders |
| T-024 | Encoder Training and Validation | 🟠 High | L | 2 – Encoders |
| T-030 | Variational Transformer Architecture | 🔴 Critical | L | 3 – Temporal |
| T-031 | ELBO and CRPS Losses | 🔴 Critical | M | 3 – Temporal |
| T-032 | Monte Carlo Inference | 🟠 High | M | 3 – Temporal |
| T-033 | Temporal Stage Runner | 🔴 Critical | L | 3 – Temporal |
| T-034 | Temporal Model Validation | 🟠 High | M | 3 – Temporal |
| T-040 | Conditional UNet Architecture | 🔴 Critical | L | 4 – Generator |
| T-041 | Spectral Loss | 🔵 Medium | S | 4 – Generator |
| T-042 | Generator Stage Runner | 🔴 Critical | M | 4 – Generator |
| T-043 | Generator Validation | 🟠 High | M | 4 – Generator |
| T-050 | Fusion Transformer Architecture | 🔴 Critical | L | 5 – Fusion |
| T-051 | Fusion Stage Runner | 🔴 Critical | L | 5 – Fusion |
| T-052 | Full System Integration Test | 🔴 Critical | M | 5 – Fusion |
| T-060 | Metrics Module | 🟠 High | M | 6 – Eval |
| T-061 | Calibration Analysis | 🟠 High | M | 6 – Eval |
| T-062 | Visualization Module | 🔵 Medium | L | 6 – Eval |
| T-063 | CLI Entry Points | 🟠 High | M | 6 – Eval |
| T-064 | Persistence Baseline | 🔵 Medium | S | 6 – Eval |
| T-070 | Test Suite | 🟠 High | L | 7 – Polish |
| T-071 | README and Documentation | 🟠 High | M | 7 – Polish |
| T-072 | Type Hints and Linting | 🔵 Medium | M | 7 – Polish |
| T-073 | Performance Profiling | 🟢 Low | L | 7 – Polish |
| T-074 | Experiment Tracking | 🟢 Low | M | 7 – Polish |

### Effort Summary

| Stage | Tasks | Estimated Time |
|---|---|---|
| 0 – Setup & Config | 4 tasks | 3–5 days |
| 1 – Data Pipeline | 6 tasks | 2–3 weeks |
| 2 – Spatial Encoders | 5 tasks | 2–3 weeks |
| 3 – Temporal Model | 5 tasks | 2–3 weeks |
| 4 – Generator | 4 tasks | 1–2 weeks |
| 5 – Fusion | 3 tasks | 1–2 weeks |
| 6 – Evaluation & CLI | 5 tasks | 1–2 weeks |
| 7 – Polish & Testing | 5 tasks | 2–3 weeks |
| **TOTAL** | **37 tasks** | **12–19 weeks (1 developer)** |

> GPU training time for Stages 2–5 is additional: estimate 1–3 days per stage on a single A100 80GB GPU with 2 years of GOES data.

---

## 9. Recommended Development Sequence

Follow this sequence for the smoothest development path. Each phase gates the next — do not begin a phase until the prior phase's success criterion is met.

| Phase | Name | Tasks | Time | Success Criterion |
|---|---|---|---|---|
| 1 | Foundation | T-001–T-004 | ~1 week | All tests pass; config loads correctly |
| 2 | Data Pipeline | T-010–T-015 | ~3 weeks | Zarr store loads; DataLoader shapes correct; spot-check plots look reasonable |
| 3 | Model Building Blocks | T-020–T-022 | ~1.5 weeks | Shape tests pass; gradients flow through all blocks |
| 4 | Stage 1 Training | T-023–T-024 | ~2 weeks | SSIM > 0.85 on reconstruction; latent t-SNE shows structure |
| 5 | Stage 2 Training | T-030–T-034 | ~2.5 weeks | CRPS improves over persistence; calibration coverage ≈ 0.9 |
| 6 | Stage 3 Training | T-040–T-043 | ~2 weeks | Generated fields are visually plausible; SSIM > 0.80 |
| 7 | Stage 4 Training | T-050–T-052 | ~2 weeks | Holistic forecast improves on individual component forecasts |
| 8 | Evaluation | T-060–T-064 | ~2 weeks | All metrics computed at all lead times; calibration diagrams generated |
| 9 | Polish | T-070–T-074 | ~2 weeks | > 80% test coverage; mypy clean |

### Key Risks

> ⚠️ **Data Pipeline (Phase 2)** is the highest-risk phase. GOES projection accuracy, file format edge cases, and Zarr performance can each cause significant delays. Budget extra time here and validate thoroughly before proceeding to model training.

> ⚠️ **Posterior collapse in the Variational Transformer (Phase 5)** is a common failure mode. Mitigate with free-bits regularization, KL annealing from 0 → β, and monitoring KL loss separately from reconstruction loss. If KL trends to 0, reduce β or increase `free_bits`.
