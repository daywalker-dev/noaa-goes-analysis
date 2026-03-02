# noaa-goes-analysis
LLM-driven analysis of noaa-GOES sat imaging.

**Probabilistic Earth-system forecasting from GOES-16/17/18 Level 2 satellite data.**

A modular, research-grade PyTorch framework that learns surface and sea dynamics from multi-channel GOES L2 products and produces uncertainty-aware multi-step atmospheric forecasts with synthetic satellite-field reconstruction.

```
surface + wind + humidity  →  future atmospheric state  →  reconstructed GOES L2 fields
```

---

## Why This Exists

Numerical weather prediction is computationally expensive and slow to iterate on. GOES geostationary satellites provide continuous, high-cadence L2 derived products — sea surface temperature, land surface temperature, cloud imagery, precipitable water, and derived motion winds — that together paint a rich picture of the atmospheric state every hour.

This framework trains a cascaded ML pipeline directly on those L2 products to learn a fast, probabilistic emulator: given the last 12 hours of observed satellite fields, forecast the next 24 hours with calibrated uncertainty at every step.

---

## Architecture

The system trains in four sequential stages:

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1 · SPATIAL CNN ENCODERS                                 │
│  Three domain-specific CNNs (land, sea, cloud) with residual    │
│  blocks and CBAM attention. Each produces a latent embedding    │
│  and a next-step reconstruction.                                │
└────────────────────────────┬────────────────────────────────────┘
                             │ latent embeddings [B, T, D]
┌────────────────────────────▼────────────────────────────────────┐
│  STAGE 2 · VARIATIONAL TRANSFORMER                              │
│  Fuses spatial latents with wind, humidity, temperature, and    │
│  pressure. A VAE bottleneck produces a calibrated distribution  │
│  over future atmospheric states: μ and σ² per forecast step.   │
└────────────────────────────┬────────────────────────────────────┘
                             │ predicted state distribution
┌────────────────────────────▼────────────────────────────────────┐
│  STAGE 3 · CONDITIONAL UNET                                     │
│  Maps predicted atmospheric state vectors back to synthetic     │
│  GOES L2 fields. State conditioning via FiLM at the bottleneck. │
└────────────────────────────┬────────────────────────────────────┘
                             │ synthetic L2 fields
┌────────────────────────────▼────────────────────────────────────┐
│  STAGE 4 · FUSION TRANSFORMER                                   │
│  Combines original L2, CNN predictions, Bayesian outputs, and   │
│  meteorological fields into a unified holistic forecast with    │
│  spatial uncertainty maps.                                      │
└─────────────────────────────────────────────────────────────────┘
```

Full architecture specification: [`PROJECT_SPEC.md`](PROJECT_SPEC.md)

---

## GOES L2 Data Products

| Product | Description |
|---|---|
| `ABI-L2-SSTF` | Sea surface temperature |
| `ABI-L2-LSTF` | Land surface temperature |
| `ABI-L2-CMIPF` | Cloud and moisture imagery (bands 7–16) |
| `ABI-L2-DMWF` | Derived motion winds (speed, direction, pressure) |
| `ABI-L2-TPWF` | Total precipitable water |
| `ABI-L2-AODF` | Aerosol optical depth *(optional)* |

Data is fetched via [`goes2go`](https://github.com/blaylockbk/goes2go), reprojected to a common lat/lon grid, aligned to a 1-hourly time axis, and stored in chunked Zarr format.

---

## Installation

```bash
git clone https://github.com/your-org/goes-forecast
cd goes-forecast
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Core dependencies:** `torch>=2.2`, `goes2go`, `xarray`, `zarr`, `pyresample`, `pyproj`

**Optional:** `pytorch-msssim`, `properscoring`, `wandb`, `cmocean`

See [`requirements.txt`](requirements.txt) for the full pinned dependency list.

---

## Quick Start

### 1. Download data

```bash
python scripts/download_data.py \
  --satellite GOES-16 \
  --start 2023-01-01 \
  --end   2023-12-31 \
  --products SST LST CMIP DMW TPW \
  --output-dir data/raw
```

### 2. Preprocess to Zarr

```bash
python scripts/preprocess.py \
  --input  data/raw \
  --output data/processed/goes_l2.zarr \
  --config config/base_config.yaml
```

### 3. Train all stages

```bash
# Stage 1 — Spatial encoders
python scripts/train.py --stage encoders --config config/base_config.yaml

# Stage 2 — Variational transformer (encoders frozen)
python scripts/train.py --stage temporal --config config/base_config.yaml

# Stage 3 — Reverse generator
python scripts/train.py --stage generator --config config/base_config.yaml

# Stage 4 — Fusion model
python scripts/train.py --stage fusion --config config/base_config.yaml

# Or run all stages sequentially
python scripts/train.py --stage all --config config/base_config.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint outputs/fusion/checkpoints/best.ckpt \
  --split test \
  --lead-times 1 3 6 12 24 \
  --n-samples 50 \
  --output-dir outputs/evaluation
```

---

## Configuration

All hyperparameters are driven by `config/base_config.yaml`. Override any value from the CLI without editing the file:

```bash
python scripts/train.py \
  --stage encoders \
  --config config/base_config.yaml \
  --override model.encoder.latent_dim=256 training.batch_size=8
```

Key config sections:

```yaml
data:
  temporal:
    input_steps: 12        # hours of input context
    forecast_steps: 24     # hours to predict

model:
  encoder:
    latent_dim: 128
    n_res_blocks: 4
  temporal:
    d_model: 256
    n_heads: 8
    beta_kl: 1.0

training:
  batch_size: 4
  mixed_precision: true
  gradient_clip: 1.0
```

---

## Project Layout

```
goes_forecast/
├── config/              YAML configs (base, encoder, temporal, experiment)
├── data/                Downloader, preprocessor, dataset, Zarr utilities
├── models/              blocks, spatial_encoder, temporal_bayesian,
│                        reverse_generator, fusion
├── training/            losses, trainer, stage_runners, callbacks
├── evaluation/          metrics, calibration, visualizer
├── utils/               config_loader, logger, reproducibility, projection
└── scripts/             download_data, preprocess, train, evaluate
```

---

## Evaluation Metrics

| Metric | Type |
|---|---|
| RMSE / MAE / Bias | Deterministic accuracy |
| SSIM | Spatial structural quality |
| Anomaly Correlation | Spatial pattern fidelity |
| CRPS | Probabilistic skill (accuracy + sharpness) |
| Coverage (90% CI) | Calibration — should be ≈ 0.9 |
| Reliability diagram | Full calibration curve |
| Multi-step degradation | Skill vs forecast lead time |

Evaluation outputs include per-variable skill curves at 1h, 3h, 6h, 12h, and 24h lead times, reliability diagrams, spatial error maps, and a comparison against the persistence baseline.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Separate domain encoders (land/sea/cloud) | Different physical regimes benefit from domain-specific feature learning |
| VAE bottleneck in the temporal model | Principled, differentiable uncertainty — posterior collapses are diagnosable and fixable |
| FiLM conditioning in the generator | More stable than concatenation; preserves spatial structure while injecting state information |
| Staged training (freeze-then-train) | Prevents gradient conflicts; each component can be independently debugged and validated |
| Zarr storage with Blosc compression | Chunked cloud-native format; efficient sequential time-window access during training |
| Physics constraint losses | Steers predictions toward physically plausible trajectories without hard constraints |

---

## Extending the Framework

**Add a new data product:** Implement `BaseIngester` in `data/downloader.py`, add the product spec to `config/base_config.yaml`, and map it to the appropriate domain encoder or meteorological field.

**Add a new encoder domain:** Subclass `SpatialCNNEncoder` in `models/spatial_encoder.py` and register it in `DomainEncoderEnsemble`.

**Add a new loss:** Implement it in `training/losses.py`, inherit from `nn.Module`, and reference it by name in the stage config.

**Add a new metric:** Implement it as a plain function in `evaluation/metrics.py`. The evaluator discovers metrics automatically from the config's `metrics` list.

---

## Reproducibility

Every training run:
- Sets global seeds for `random`, `numpy`, `torch`, and CUDA at startup
- Saves a snapshot of the full resolved config alongside the checkpoint
- Assigns a unique experiment ID (timestamp + hash) to each run
- Logs the Python and library versions used

To reproduce a run from a checkpoint:

```bash
python scripts/train.py \
  --config outputs/fusion/run_20240315_abc123/config.yaml \
  --resume outputs/fusion/checkpoints/best.ckpt
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 16 GB | 40–80 GB (A100) |
| CPU RAM | 32 GB | 64+ GB |
| Disk (1 year of data) | ~500 GB | 2+ TB NVMe |
| Training time per stage | — | 1–3 days on A100 |

Mixed precision (bfloat16) is enabled by default and required for the recommended batch sizes on < 80 GB GPUs.

---

## Citation

If you use this framework in research, please cite the GOES-16 instrument paper and goes2go:

```
Blaylock, B.K. (2022). goes2go: Download and display GOES-East and GOES-West data.
Zenodo. https://doi.org/10.5281/zenodo.6654485
```

---

## License

MIT — see [`LICENSE`](LICENSE) for details.