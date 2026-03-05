"""
main.py — GOES L2 Probabilistic Forecast Framework
====================================================
Entry point for all pipeline stages:

    python main.py download   --config config/default.yaml
    python main.py preprocess --config config/default.yaml
    python main.py train      --config config/default.yaml --stage encoders
    python main.py train      --config config/default.yaml --stage temporal
    python main.py train      --config config/default.yaml --stage decoder
    python main.py train      --config config/default.yaml --stage fusion
    python main.py evaluate   --config config/default.yaml --checkpoint checkpoints/fusion_best.pt
    python main.py forecast   --config config/default.yaml --checkpoint checkpoints/fusion_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import load_config
from utils.logging import get_logger

logger = get_logger("main")

VALID_STAGES = ["encoders", "temporal", "decoder", "fusion"]


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    """Download GOES L2 products for the configured date range."""
    from data.ingest import GOESDataIngester

    cfg = load_config(args.config)
    logger.info(f"Starting download | satellite=GOES-{cfg.data.goes_satellite}")

    products = [p.to_dict() if hasattr(p, "to_dict") else p for p in cfg.data.products]
    ingester = GOESDataIngester(
        satellite=cfg.data.goes_satellite,
        raw_dir=cfg.data.raw_dir,
        products=products,
    )

    files = ingester.download_range(
        start=cfg.data.date_range.start,
        end=cfg.data.date_range.end,
    )

    total = sum(len(v) for v in files.values())
    logger.info(f"Download complete — {total} files across {len(files)} products")
    for product, flist in files.items():
        logger.info(f"  {product}: {len(flist)} files")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Reproject, normalize, and write aligned Zarr store."""
    from data.ingest import GOESDataIngester
    from data.preprocess import GOESPreprocessor

    cfg = load_config(args.config)
    logger.info("Starting preprocessing pipeline")

    # Collect already-downloaded files
    ingester = GOESDataIngester(
        satellite=cfg.data.goes_satellite,
        raw_dir=cfg.data.raw_dir,
        products=[p.to_dict() if hasattr(p, "to_dict") else p for p in cfg.data.products],
    )
    files_by_product = {
        p.name if hasattr(p, "name") else p["name"]: ingester.get_cached_files(
            p.name if hasattr(p, "name") else p["name"]
        )
        for p in cfg.data.products
    }

    preprocessor = GOESPreprocessor(
        zarr_store=cfg.data.zarr_store,
        grid_size=list(cfg.data.spatial.grid_size),
        lat_bounds=list(cfg.data.spatial.lat_bounds),
        lon_bounds=list(cfg.data.spatial.lon_bounds),
        normalization=cfg.data.normalization.method,
        window_size=cfg.data.temporal.window_size,
        forecast_horizon=cfg.data.temporal.forecast_horizon,
        step_hours=cfg.data.temporal.step_hours,
    )

    store_path = preprocessor.process_files(files_by_product)
    logger.info(f"Zarr store ready: {store_path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Run a training stage."""
    import torch
    from training.trainer import StageTrainer

    cfg = load_config(args.config)
    stage = args.stage

    if stage not in VALID_STAGES:
        logger.error(f"Invalid stage '{stage}'. Choose from: {VALID_STAGES}")
        sys.exit(1)

    # Find stage config
    stage_cfg = next(
        (s for s in cfg.training.stages if s.name == stage), None
    )
    if stage_cfg is None:
        logger.error(f"Stage '{stage}' not found in config training.stages")
        sys.exit(1)

    device = _resolve_device(cfg.project.device)
    logger.info(f"Training stage='{stage}' | device={device}")

    _set_seed(cfg.project.seed)

    trainer = StageTrainer(
        cfg=cfg,
        stage=stage,
        stage_cfg=stage_cfg,
        device=device,
        checkpoint_dir=cfg.training.save_dir,
        log_dir=cfg.training.log_dir,
    )
    trainer.run()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model checkpoint on the test split."""
    import torch
    from evaluation.evaluator import Evaluator

    cfg = load_config(args.config)
    device = _resolve_device(cfg.project.device)
    logger.info(f"Evaluating checkpoint: {args.checkpoint}")

    evaluator = Evaluator(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        device=device,
        output_dir=cfg.evaluation.output_dir,
    )
    results = evaluator.run()

    logger.info("=== Evaluation Results ===")
    for metric, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")


def cmd_forecast(args: argparse.Namespace) -> None:
    """Run inference and produce forecast outputs."""
    import torch
    from evaluation.evaluator import Evaluator

    cfg = load_config(args.config)
    device = _resolve_device(cfg.project.device)
    logger.info(f"Running forecast | checkpoint={args.checkpoint}")

    evaluator = Evaluator(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        device=device,
        output_dir=cfg.evaluation.output_dir,
    )
    evaluator.forecast(
        n_samples=getattr(args, "n_samples", 50),
        save_outputs=True,
    )
    logger.info(f"Forecast outputs saved to: {cfg.evaluation.output_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> str:
    """Fall back to CPU if CUDA requested but unavailable."""
    import torch
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        return "cpu"
    return device_str


def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goes_forecast",
        description="GOES L2 Probabilistic Earth-System Forecasting Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # download
    p_dl = sub.add_parser("download", help="Download GOES L2 products via goes2go")
    p_dl.set_defaults(func=cmd_download)

    # preprocess
    p_pre = sub.add_parser("preprocess", help="Reproject + normalize → Zarr store")
    p_pre.set_defaults(func=cmd_preprocess)

    # train
    p_train = sub.add_parser("train", help="Train a pipeline stage")
    p_train.add_argument(
        "--stage", "-s",
        required=True,
        choices=VALID_STAGES,
        help="Which stage to train",
    )
    p_train.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Resume training from checkpoint path",
    )
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model on test split")
    p_eval.add_argument(
        "--checkpoint", "-k",
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    # forecast
    p_fc = sub.add_parser("forecast", help="Run probabilistic forecast inference")
    p_fc.add_argument(
        "--checkpoint", "-k",
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    p_fc.add_argument(
        "--n-samples",
        type=int,
        default=50,
        dest="n_samples",
        help="Monte Carlo samples for uncertainty estimation (default: 50)",
    )
    p_fc.set_defaults(func=cmd_forecast)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()