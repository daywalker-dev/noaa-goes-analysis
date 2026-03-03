"""Evaluation framework: metrics, calibration, visualization."""
from goes_forecast.evaluation.metrics import (
    rmse, mae, bias, ssim_score, spatial_correlation,
    crps_gaussian, crps_ensemble, coverage_score, multistep_skill,
    get_metric, METRIC_REGISTRY,
)
from goes_forecast.evaluation.calibration import (
    reliability_diagram, rank_histogram, sharpness, calibration_summary,
)
