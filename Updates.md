# UPDATES — Needed Improvements & Next Steps

This document lists the updates, enhancements, and known gaps remaining in the GOES L2 probabilistic forecasting framework. Items are prioritised by urgency.

---

## Critical (Must-Fix Before Production Runs)

### 1. `goes2go` API Compatibility Verification

The `data/ingest.py` download functions use the `goes2go.GOES` class with `.nearesttime()`. The exact API surface of `goes2go` varies between versions, and some L2 products (especially `ABI-L2-DMWP` and `ABI-L2-ADPP`) may not be directly downloadable with the same interface or may require different domain codes. Before running real data pipelines, verify that every product in `default.yaml` is actually available via `goes2go` for the selected satellite and domain, and update the `download_goes_product()` function if the API differs. Some products return multi-band datasets where the variable names do not match the keys in `PRODUCT_VARIABLE_MAP` — audit these mappings against actual downloaded NetCDFs.

### 2. Spatial Reprojection Precision

The current `reproject_to_common_grid()` function uses `xr.Dataset.interp()` with linear interpolation. GOES ABI data is delivered in a geostationary fixed-grid coordinate system (`x`/`y` in radians), not regular lat/lon. For research-grade fidelity, replace with `pyresample` or `satpy` reprojection that properly handles the ABI fixed-grid → equirectangular transformation and accounts for Earth curvature, terrain, and scan geometry. The current approach will introduce systematic spatial distortion.

### 3. Temporal Alignment Robustness

The `align_temporal()` function assumes all datasets share a `t` dimension that can be resampled. In practice, different GOES L2 products are generated at different cadences (e.g., SST hourly, DMW every 10 min). The concatenation and resampling logic needs per-product cadence handling and more robust tolerance windows. Add unit tests with real multi-cadence datasets.

### 4. Fusion Stage Training — Dynamic Projection Layers

In `training/trainer.py`, the `train_fusion()` function creates ad-hoc `nn.Linear` projection layers stored as function attributes (`train_fusion._orig_proj`, etc.). This is a prototype shortcut. These projections must be moved into the fusion model class itself or into a dedicated `FusionInputAdapter` module so they are properly tracked by the optimizer, saved in checkpoints, and restored on reload.

---

## High Priority (Required for Reliable Results)

### 5. Encoder–Decoder Spatial Size Consistency

The spatial encoders downsample inputs through `DownBlock` layers, producing feature maps at reduced resolution. The prediction head outputs at this reduced resolution, but training compares against targets at full resolution. Add explicit `F.interpolate` in the training prep function, or change the encoder prediction head to include upsampling layers. Without this fix, training will either crash on size mismatch or silently learn at the wrong resolution.

### 6. Multi-Step Autoregressive Evaluation Script

The `evaluate.py` script contains a `multistep_degradation()` call but does not wire it to real data yet. Implement the full evaluation loop that: (a) loads test data, (b) runs the encoder pipeline to produce initial latent states, (c) calls `temporal_model.forecast()` for each sample, (d) decodes each forecast step through the reverse generator, and (e) computes metrics at each horizon. This is the most important evaluation mode for a forecasting system.

### 7. CRPS Ensemble Estimator Efficiency

The `crps_ensemble()` function uses a double for-loop over sample pairs, which is O(S²) and will be very slow for large ensembles. Replace with the sorted-ensemble PWM estimator which is O(S log S): sort the ensemble, then compute `(2i - 1) * x_sorted[i]` sums.

### 8. Checkpoint–Config Consistency

Currently, each training stage saves only model weights. The training engine should also save: the optimizer state, the scheduler state, the epoch number, the current validation loss, and a hash of the config. This allows proper resumption and prevents accidentally loading weights that were trained with a different config.

---

## Medium Priority (Quality & Completeness)

### 9. Physics-Aware Loss Expansion

The `PhysicsAwareLoss` in `models/blocks.py` penalises Laplacian roughness and temporal jumps. Add domain-specific constraints such as: SST spatial gradient penalties near coastlines (SST should not have sharp discontinuities over open water), conservation of total precipitable water in neighbouring time-steps (moisture budget), and wind field divergence / vorticity constraints derived from the Navier-Stokes equations.

### 10. Proper Train/Val/Test Splits

The current `build_dataloaders()` function splits data by a single temporal cut. For rigorous evaluation, implement a three-way split (train / validation / held-out test) with configurable ratios, and optionally support k-fold temporal cross-validation where each fold uses a contiguous block as the test set.

### 11. Data Augmentation

Add spatial augmentation (random crops, flips, rotations) and temporal augmentation (jittered windows, random dropouts of input steps) to the dataset class. These should be configurable via the YAML and disabled at validation/test time.

### 12. Logging Integration

Replace print/logger statements with a proper experiment tracker. Add a `utils/logging.py` module that wraps either TensorBoard, Weights & Biases, or MLflow. Log: per-epoch train/val losses, learning rate, gradient norms, sample predictions, and evaluation metrics. Make the backend configurable.

### 13. Distributed Training Support

The training engine currently assumes single-GPU. Add support for `torch.nn.parallel.DistributedDataParallel` with proper gradient synchronisation, or integrate with PyTorch Lightning / Hugging Face Accelerate for multi-GPU and multi-node scaling.

### 14. Missing Data-Aware Loss Masking

The `CombinedLoss` accepts a mask but simply zeros out invalid pixels before computing MSE/SSIM. SSIM computed on partially masked fields is unreliable because the sliding-window kernel spans valid and invalid regions. Implement mask-aware SSIM that only considers windows with sufficient valid coverage (e.g., >80% valid).

---

## Low Priority (Future Enhancements)

### 15. Additional Satellite Support

The framework currently targets GOES-16/17/18 via `goes2go`. Add adapters for Himawari-8/9 (via `satpy`), Meteosat (EUMETSAT API), and polar-orbiting instruments (MODIS, VIIRS). The encoder architecture is channel-count-agnostic, so the main work is in the data ingestion layer.

### 16. Probabilistic Decoder

The reverse generator currently produces a single deterministic reconstruction. Extend it to output mean + variance (or a full covariance) per pixel, enabling probabilistic field reconstructions and uncertainty maps of the reconstructed L2 fields.

### 17. Attention Visualisation

Add hooks to extract and visualise attention maps from the spatial self-attention layers in the encoders and the cross-attention layers in the fusion model. This supports interpretability and physical plausibility checks.

### 18. ONNX / TorchScript Export

Add export scripts that convert trained models to ONNX or TorchScript for deployment in inference-only environments without Python dependencies.

### 19. Benchmark Suite

Create a benchmark script that measures: training throughput (samples/sec), inference latency (ms/sample), peak GPU memory, and model parameter counts for each stage. Store results in a JSON report for regression tracking.

---

## File-Specific Fix List

| File | Issue | Fix |
|------|-------|-----|
| `data/ingest.py` | `reproject_to_common_grid` uses naive interp | Replace with pyresample |
| `data/ingest.py` | `align_temporal` assumes shared `t` dim | Per-product cadence handling |
| `data/ingest.py` | `PRODUCT_VARIABLE_MAP` may not match real NetCDFs | Audit against actual data |
| `data/dataset.py` | Aux features truncated to 32 dims arbitrarily | Make configurable or use adaptive pooling |
| `models/blocks.py` | SSIM kernel not normalised per-channel for varying sizes | Add input-size guards |
| `models/encoders/spatial_cnn.py` | Pred head at reduced resolution | Add upsampling or match target |
| `models/temporal/probabilistic.py` | `UncertaintyTransformer` PE may mishandle odd dims | Floor division safety |
| `models/decoder/reverse_generator.py` | Skip connections are self-referencing (not a true UNet) | Wire skip from matching encoder |
| `models/fusion/fusion_model.py` | `ConcatMLPFusion` and `FiLMFusion` accept but ignore `num_heads`/`num_layers` kwargs | Clean up signatures |
| `training/trainer.py` | `train_fusion` uses function-level attrs for projections | Move to model class |
| `training/trainer.py` | No gradient clipping | Add configurable `max_grad_norm` |
| `evaluation/metrics.py` | `crps_ensemble` is O(S²) | Use sorted PWM estimator |
| `scripts/evaluate.py` | Real data path not implemented | Wire to data pipeline |

---

## Scripts Needed

### `scripts/download_data.py`

A standalone script to download and preprocess GOES L2 data without training. Should accept the same config file and produce Zarr stores ready for training.

```
python -m goes_forecast.scripts.download_data --config configs/default.yaml
```

### `scripts/export_model.py`

Export trained models to ONNX or TorchScript.

```
python -m goes_forecast.scripts.export_model --config configs/default.yaml --format onnx
```

### `scripts/visualise_predictions.py`

Generate side-by-side plots of predicted vs. observed L2 fields at each forecast horizon, with uncertainty overlays.

```
python -m goes_forecast.scripts.visualise_predictions --config configs/default.yaml --sample-idx 0
```