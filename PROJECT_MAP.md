# GOES L2 Probabilistic Forecast Framework — Complete Project Map

## 1. File Inventory & Method Map

### Entry Points

#### `main.py`
- `cmd_download(args)` — Download GOES L2 products for configured date range
- `cmd_preprocess(args)` — Reproject, normalize, write aligned Zarr store
- `cmd_train(args)` — Run a training stage (encoders/temporal/decoder/fusion)
- `cmd_evaluate(args)` — Evaluate a trained model checkpoint on test split
- `cmd_forecast(args)` — Run probabilistic forecast inference
- `_resolve_device(device_str)` — Resolve CUDA/CPU device
- `_set_seed(seed)` — Set global RNG seeds
- `build_parser()` — Build argparse CLI
- `main()` — Entry point

#### `scripts/train.py`
- `_make_synthetic_data(cfg)` — Generate synthetic data for dry-run
- `main()` — Alternate training entry point (stages 1–4 sequentially)

#### `scripts/evaluate.py`
- `PersistenceBaseline.predict(input_data, n_steps)` — Trivial repeat-last baseline
- `_get_domain_channels(cfg)` — Extract domain→channel-count map
- `_get_domain_indices(cfg)` — Extract domain→channel-index map
- `_total_channels(cfg)` — Sum all product channels
- `_all_channel_names(cfg)` — List all channel names
- `load_models(cfg, checkpoint_dir, device)` — Load all 4 stages from checkpoints

#### `scripts/download_data.py`
- `main()` — CLI download entry point (click-based)

#### `scripts/preprocess.py`
- `main()` — CLI preprocessing entry point (click-based)

---

### Data Pipeline (`data/`)

#### `data/__init__.py`
- Re-exports: `GOESWindowDataset`, `build_dataloader`, `ZarrStore`, `GOESDownloader`, `GOESPreprocessor`

#### `data/downloader.py` — `GOESDownloader`
- `__init__(satellite, output_dir, products)`
- `_resolve_product(shortname)` — Map short name → ABI product ID
- `download(start, end, max_retries)` → `pd.DataFrame` manifest
- `build_manifest()` — Scan existing files on disk

#### `data/preprocessor.py` — `GOESPreprocessor`
- `__init__(cfg: DictConfig)` — Reads preprocessing config section
- `_parse_goes_file(path)` — Extract data + DQF from NetCDF
- `_compute_quality_mask(dqf)` — Build boolean quality mask
- `_reproject_scene(data, x, y, proj_info)` — Reproject via pyresample or scipy fallback
- `_resize_to_grid(data)` — Resize to target grid shape
- `_fill_missing(data, mask)` — Gap-fill missing values
- `process_files(file_list, product_id)` → `list[dict]` with data/mask/product
- `compute_stats(data_dict)` — Per-channel mean/std
- `normalize(data, means, stds)` — Z-score normalization
- `write_zarr(output_path, data, channel_names, times, stats)` — Write chunked Zarr store

#### `data/zarr_store.py` — `ZarrStore`
- `__init__(zarr_path, stats_path)`
- `shape` property → `(T, C, H, W)`
- `get_window(t_start, t_end)` → temporal window
- `get_patch(t_start, t_end, lat_start, lat_end, lon_start, lon_end)` → spatio-temporal patch
- `get_channel_indices(domain, products)` → channel index list
- `get_stats_arrays()` → `(means, stds)` shaped `(C,1,1)`
- `time_index_for_date(date_str)` → closest time index

#### `data/dataset.py` — `GOESWindowDataset(Dataset)`
- `__init__(zarr_path, stats_path, cfg, split, transform)`
- `_build_indices()` — Build valid `(time, lat, lon)` start indices
- `__len__()` / `__getitem__(idx)` → dict with input/target/mask/meteo
- `build_dataloader(cfg, split)` — Factory for DataLoader

#### `data/augmentation.py`
- `RandomFlip(horizontal, vertical, p)` — Random flip transform
- `TemporalJitter(max_offset)` — Random time offset
- `Compose(transforms)` — Chain transforms
- `build_augmentation(cfg)` → `Compose | None`

---

### Models (`models/`)

#### `models/__init__.py`
- Re-exports from `blocks`, `spatial_encoder`, `temporal_bayesian`, `reverse_generator`, `fusion`

#### `models/blocks.py`
- `ResidualBlock(channels, dropout)` — Pre-activation residual block
- `DownBlock(in_ch, out_ch, dropout)` — Strided conv downsample + residual
- `UpBlock(in_ch, out_ch, skip_ch, dropout)` — Transposed conv upsample + residual + skip
- `SpatialSelfAttention(channels, num_heads)` — Multi-head self-attention over H×W
- `ChannelAttention(channels)` — SE-style channel attention
- `SpatialAttention()` — Max+avg pool spatial attention
- `CBAM(channels)` — Channel + spatial attention module
- `FiLM(cond_dim, channels)` — Feature-wise linear modulation
- `SinusoidalPE(d_model)` — Sinusoidal positional encoding
- `ssim_loss(pred, target, window_size)` — Differentiable SSIM loss function
- `PhysicsAwareLoss` — Laplacian + temporal gradient regularization
- `CombinedLoss(mse_weight, ssim_weight, physics_weight)` — Weighted loss combo

#### `models/spatial_encoder.py`
- `SpatialCNNEncoder(in_channels, latent_dim, base_channels, channel_multipliers, n_res_blocks, use_cbam, dropout)`
  - `encode(x)` → `(latent, skips)`
  - `decode(latent, skips, target_size)` → reconstruction
  - `forward(x)` → `{"latent", "reconstruction"}`
  - `encode_only(x)` → latent vector
- `DomainEncoderEnsemble(domain_channels, encoder_cfg)`
  - `forward(domain_inputs)` → `{"latents", "latent_<domain>", "recon_<domain>"}`
  - `encode_only(domain_inputs)` → concatenated latents
  - `freeze()` / `unfreeze()`

#### `models/temporal_bayesian.py` — `VariationalTransformer`
- `__init__(latent_dim, meteo_dim, state_dim, d_model, n_heads, ...)`
- `_reparameterize(mu, logvar)` → sampled z
- `_compute_kl(mu, logvar)` → KL divergence with free-bits
- `forward(spatial_latents, meteo_fields)` → `{"mu", "logvar", "kl_loss", "z"}`
- `sample(spatial_latents, meteo_fields, n_samples)` → `{"mean", "std", "p05", "p95"}`
- (Also contains `MeteoProjector` helper class)

#### `models/reverse_generator.py` — `ConditionalUNet`
- `__init__(state_dim, out_channels, noise_dim, base_channels, channel_multipliers, ...)`
- `forward(state, noise, target_size)` → `(B, out_channels, H, W)`
- `decode_sequence(states, target_size)` → `(B, T, out_channels, H, W)`

#### `models/fusion.py` — `FusionTransformer`
- `__init__(cnn_dim, bayes_dim, meteo_dim, gen_dim, d_model, n_heads, ...)`
- `forward(cnn_latents, bayes_output, meteo_fields, gen_features, gen_spatial)` → `{"forecast", "uncertainty", "scales", "fused_features"}`

---

### Training (`training/`)

#### `training/__init__.py`
- Re-exports: `MaskedMSE`, `SSIMLoss`, `CRPSLoss`, `SpectralLoss`, `PhysicsConstraintLoss`, `ELBOLoss`, `CompositeLoss`, `BaseTrainer`, `EncoderStageRunner`, `TemporalStageRunner`, `GeneratorStageRunner`, `FusionStageRunner`

#### `training/losses.py`
- `MaskedMSE` — MSE on valid pixels only
- `SSIMLoss(data_range, win_size)` — 1 − SSIM loss
- `CRPSLoss` — Closed-form Gaussian CRPS
- `SpectralLoss` — L2 distance between 2D FFT magnitudes
- `PhysicsConstraintLoss(temporal_weight, spatial_weight, energy_weight)` — Physics regularization
- `ELBOLoss(beta, free_bits)` — Reconstruction + β·KL with annealing via `set_beta()`
- `CompositeLoss(components, weights)` — Weighted sum of named losses

#### `training/trainer.py`
- `_build_scheduler(optimizer, cfg, total_epochs)` — Scheduler factory
- `_run_epoch(model, loader, criterion, optimizer, ...)` — Generic epoch runner
- `BaseTrainer` — Base class for stage-specific trainers
- `train_encoder(encoder, train_loader, val_loader, cfg, device, tag)` — Stage 1 functional API
- `train_temporal(...)` — Stage 2 functional API
- `train_decoder(...)` — Stage 3 functional API
- `train_fusion(...)` — Stage 4 functional API
- `_EncoderWrapper` — Thin wrapper for generic epoch runner
- `CombinedLoss` — Also defined here (duplicate of blocks.py version)

#### `training/stage_runners.py`
- `_split_domains(batch, domain_indices)` — Split input by domain
- `EncoderStageRunner(BaseTrainer)` — Stage 1: DomainEncoderEnsemble training
- `TemporalStageRunner(BaseTrainer)` — Stage 2: VariationalTransformer with frozen encoders
- `GeneratorStageRunner(BaseTrainer)` — Stage 3: ConditionalUNet with frozen encoder+temporal
- `FusionStageRunner(BaseTrainer)` — Stage 4: FusionTransformer, optionally unfreezes all

---

### Evaluation (`evaluation/`)

#### `evaluation/__init__.py`
- Re-exports from `metrics` and `calibration`

#### `evaluation/metrics.py`
- `rmse(pred, target, mask)`, `mae(pred, target, mask)`, `bias(pred, target, mask)`
- `ssim_score(pred, target)` / `ssim_metric(pred, target, window_size)`
- `spatial_correlation(pred, target)` — Pearson correlation over spatial dims
- `crps_gaussian(mu, sigma, obs)` — Closed-form Gaussian CRPS
- `crps_ensemble(samples, obs)` — Ensemble-based CRPS
- `coverage_score(...)` — Prediction interval coverage
- `multistep_skill(...)` — Skill degradation across horizons
- `get_metric(name)` / `METRIC_REGISTRY` — Metric lookup
- `save_evaluation_plots(results, plot_dir)` — Generate matplotlib plots

#### `evaluation/calibration.py`
- `reliability_diagram(...)`, `rank_histogram(...)`, `sharpness(...)`, `calibration_summary(...)`

#### `evaluation/visualizer.py` (referenced but may not exist as separate file)
- `plot_forecast_map(...)`, `plot_uncertainty_bands(...)`, `plot_skill_curves(...)`, `plot_reliability_diagram(...)`, `plot_spatial_error(...)`

---

### Utilities (`utils/`)

#### `utils/__init__.py`
- Re-exports: `load_config`, `validate_config`, `save_config`, `get_logger`, `set_global_seed`, `set_deterministic`, `get_environment_info`

#### `utils/config_loader.py`
- `EncoderConfig`, `TemporalConfig`, `GeneratorConfig`, `FusionModelConfig` — Pydantic schemas
- `load_config(config_path, overrides)` → `DictConfig`
- `validate_config(cfg)` — Validate with Pydantic
- `generate_experiment_id(cfg)` → string
- `save_config(cfg, path)` — Write YAML
- `get_stage_config(cfg, stage)` → stage-specific config section

#### `utils/logger.py`
- `get_logger(name, level, log_file)` → `logging.Logger` (with optional Rich handler)

#### `utils/reproducibility.py`
- `set_global_seed(seed)` — Seed random/numpy/torch/CUDA
- `set_deterministic(enabled)` — Toggle CUDNN determinism
- `get_environment_info()` → dict of versions/GPU info
- `EarlyStopping(patience)` — (referenced in tests, may exist)
- `amp_context(...)` — (referenced in tests, may exist)

#### `utils/projection.py`
- `compute_grid_coords(lat_range, lon_range, resolution_deg)` → `(lats, lons)`
- `goes_fixed_grid_to_latlon(x, y, proj_info)` — Convert GOES fixed grid to lat/lon
- `make_target_area(lat_range, lon_range, resolution)` — Build pyresample target area
- `reproject_to_grid(data, src_lats, src_lons, target_area)` — Reproject using pyresample

---

### Config Files

#### `config/default.yaml` — Training-oriented config
- Flat product list (strings): `["ABI-L2-SSTP", ...]`
- Training stages: `stage_1_encoder`, `stage_2_temporal`, etc.
- Encoder/temporal/decoder/fusion model sections at top level

#### `config/base_config.yaml` — OmegaConf-oriented config
- Structured product list (dicts): `[{id, domain, channels, required}, ...]`
- Training stages: `encoders`, `temporal`, `generator`, `fusion` (nested under `training.stages`)
- Model sections nested under `model.*`

---

## 2. Import Errors Found & Fixes

### `main.py`
| Line | Broken Import | Fix |
|------|--------------|-----|
| `cmd_download` | `from data.ingest import GOESDataIngester` | `from data.downloader import GOESDownloader` |
| `cmd_download` | `cfg.data.goes_satellite` | `cfg.data.satellite` |
| `cmd_preprocess` | `from data.ingest import GOESDataIngester` | `from data.downloader import GOESDownloader` |
| `cmd_preprocess` | `from data.preprocess import GOESPreprocessor` | `from data.preprocessor import GOESPreprocessor` |
| `cmd_preprocess` | `GOESPreprocessor(zarr_store=..., grid_size=..., ...)` | `GOESPreprocessor(cfg)` — takes DictConfig only |
| `cmd_preprocess` | `cfg.data.goes_satellite`, `cfg.data.raw_dir`, `cfg.data.zarr_store` | Use `cfg.data.satellite`, `cfg.data.raw_dir`, `cfg.data.zarr_path` |
| `cmd_train` | `from training.trainer import StageTrainer` | `from training.stage_runners import EncoderStageRunner, ...` |

### `scripts/train.py`
| Line | Broken Import | Fix |
|------|--------------|-----|
| top | `from utils.config import load_config, validate_config, save_config` | `from utils.config_loader import load_config, validate_config, save_config` |
| top | `from utils.reproducibility import seed_everything, get_device` | `from utils.reproducibility import set_global_seed, get_environment_info` + define `get_device` locally |
| top | `from models.encoders.spatial_cnn import build_encoder` | `from models.spatial_encoder import SpatialCNNEncoder` |
| top | `from models.temporal.probabilistic import build_temporal_model` | `from models.temporal_bayesian import VariationalTransformer` |
| top | `from models.decoder.reverse_generator import build_decoder` | `from models.reverse_generator import ConditionalUNet` |
| top | `from models.fusion.fusion_model import build_fusion_model` | `from models.fusion import FusionTransformer` |
| top | `from training.trainer import train_encoder, train_temporal, train_decoder, train_fusion` | `from training.stage_runners import EncoderStageRunner, TemporalStageRunner, GeneratorStageRunner, FusionStageRunner` |
| data | `from data.ingest import download_all_products, reproject_to_common_grid, align_temporal, FieldNormalizer, fill_missing` | Remove entirely — use `GOESDownloader` + `GOESPreprocessor` |
| data | `from data.dataset import build_dataloaders` | `from data.dataset import build_dataloader` (singular) |
| data | `from data.ingest import PRODUCT_VARIABLE_MAP` | Remove — doesn't exist |

### `tests/test_smoke.py`
| Line | Broken Import | Fix |
|------|--------------|-----|
| config | `from utils.config import load_config, validate_config` | `from utils.config_loader import load_config, validate_config` |
| config | `from utils.config import merge_configs, get_nested` | `from utils.config_loader import ...` (if these exist) |
| repro | `from utils.reproducibility import seed_everything` | `from utils.reproducibility import set_global_seed` |

### `models/__init__.py`
| Import | Status |
|--------|--------|
| `from models.blocks import FiLM, SinusoidalPE` | `FiLM` is in blocks.py ✓; `SinusoidalPE` may only be in `temporal_bayesian.py` — needs re-export or relocation |

### `evaluation/__init__.py`
| Import | Status |
|--------|--------|
| `from evaluation.calibration import ...` | File may not exist — create stub or full implementation |

### `__init__.py` (root)
| Import | Issue |
|--------|-------|
| `import config` | `config/` is a directory with no `__init__.py` — will fail |

---

## 3. Config Schema Divergence

The project has **two competing config files** (`config/default.yaml` and `config/base_config.yaml`) with different structures. The codebase is split — `main.py` and `scripts/train.py` assume different schemas.

**Recommendation**: Use `config/base_config.yaml` as the canonical config (it has structured product definitions with domain/channel info required by the dataset and stage runners). Rename it to `config/default.yaml` and retire the flat-list version.

---

## 4. Disk Size Management Issues

**Current problem**: `GOESDownloader.download()` fetches the entire date range in one call via `goes2go.GOES.timerange()`, which can produce hundreds of GB of raw NetCDF files that persist on disk indefinitely.

**Fixes needed**:
- Download one timestep at a time, preprocess immediately, append to Zarr, delete raw file
- Increase interval from 1 hour to 3 hours (reduces file count by 3×)
- Add configurable disk budget with monitoring
- Add cleanup of `goes2go` cache directory (`~/data/`)