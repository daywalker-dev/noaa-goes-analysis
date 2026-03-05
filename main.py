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

from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger("main")

VALID_STAGES = ["encoders", "temporal", "decoder", "fusion"]


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_download(args: argparse.Namespace) -> None:
    """Download GOES L2 products for the configured date range."""
    from data.downloader import GOESDownloader

    cfg = load_config(args.config)
    satellite = cfg.data.satellite
    logger.info(f"Starting download | satellite={satellite}")

    # Build product shortname list from config
    product_names = []
    for p in cfg.data.products:
        pid = p["id"] if isinstance(p, dict) else p
        # Extract shortname: ABI-L2-SSTF → SST
        shortname = pid.split("-")[-1].replace("F", "").replace("P", "")
        product_names.append(shortname)

    downloader = GOESDownloader(
        satellite=satellite,
        output_dir=cfg.data.raw_dir,
        products=product_names,
    )

    manifest = downloader.download(
        start=cfg.data.get("date_range", {}).get("start", "2023-06-01"),
        end=cfg.data.get("date_range", {}).get("end", "2023-08-31"),
    )

    logger.info(f"Download complete — {len(manifest)} files total")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Reproject, normalize, and write aligned Zarr store."""
    import numpy as np
    from data.preprocessor import GOESPreprocessor

    cfg = load_config(args.config)
    logger.info("Starting preprocessing pipeline")

    preprocessor = GOESPreprocessor(cfg)
    input_path = Path(cfg.data.raw_dir)

    # Process each product
    all_data = []
    channel_names = []
    for product in cfg.data.products:
        pid = product["id"] if isinstance(product, dict) else product
        shortname = pid.split("-")[-1].replace("F", "").replace("P", "")
        product_dir = input_path / shortname
        if not product_dir.exists():
            required = product.get("required", True) if isinstance(product, dict) else True
            if required:
                logger.warning(f"Required product dir missing: {product_dir}")
            continue

        files = sorted(product_dir.glob("*.nc"))
        if not files:
            continue

        results = preprocessor.process_files(files, pid)
        for r in results:
            all_data.append(r["data"])

        if isinstance(product, dict) and "channels" in product:
            channel_names.extend(product["channels"])

    if not all_data:
        logger.error("No data processed. Check input directory.")
        return

    data = np.stack(all_data, axis=0)  # (T, C, H, W)
    times = np.arange(data.shape[0]).astype("datetime64[h]") + np.datetime64("2023-01-01")

    stats = preprocessor.compute_stats({
        name: data[:, i] for i, name in enumerate(channel_names) if i < data.shape[1]
    })

    output_path = cfg.data.get("zarr_path", "data/processed/goes_l2.zarr")
    preprocessor.write_zarr(output_path, data, channel_names, times, stats)
    logger.info(f"Zarr store ready: {output_path}")


def cmd_download_and_process(args: argparse.Namespace) -> None:
    """Stream-download: download each timestep, preprocess, append to Zarr, delete raw.

    This is the disk-efficient pipeline that keeps total size under the budget.
    """
    from data.streaming_pipeline import StreamingPipeline

    cfg = load_config(args.config)
    logger.info("Starting streaming download+preprocess pipeline")

    pipeline = StreamingPipeline(cfg)
    pipeline.run()
    logger.info("Streaming pipeline complete")


def cmd_train(args: argparse.Namespace) -> None:
    """Run a training stage."""
    import torch
    from utils.config_loader import get_stage_config
    from data.dataset import build_dataloader

    cfg = load_config(args.config)
    stage = args.stage

    if stage not in VALID_STAGES:
        logger.error(f"Invalid stage '{stage}'. Choose from: {VALID_STAGES}")
        sys.exit(1)

    device = _resolve_device(cfg.project.get("device", "cuda"))
    logger.info(f"Training stage='{stage}' | device={device}")
    _set_seed(cfg.project.get("seed", 42))

    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    # Compute domain indices from config
    domain_indices = _build_domain_indices(cfg)

    if stage == "encoders":
        from training.stage_runners import EncoderStageRunner
        from models.spatial_encoder import DomainEncoderEnsemble

        domain_ch = _get_domain_channels(cfg)
        model = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
        runner = EncoderStageRunner(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            output_dir=cfg.project.get("checkpoint_dir", "checkpoints"),
            domain_indices=domain_indices,
        )
        runner.run()

    elif stage == "temporal":
        from training.stage_runners import TemporalStageRunner
        from models.spatial_encoder import DomainEncoderEnsemble
        from models.temporal_bayesian import VariationalTransformer

        domain_ch = _get_domain_channels(cfg)
        encoders = _load_encoders(cfg, domain_ch, device)
        meteo_dim = len(domain_indices.get("meteo", [])) or 1

        temporal_model = VariationalTransformer(
            latent_dim=encoders.combined_latent_dim,
            meteo_dim=meteo_dim,
            state_dim=cfg.model.temporal.state_dim,
            d_model=cfg.model.temporal.d_model,
            n_heads=cfg.model.temporal.n_heads,
            n_encoder_layers=cfg.model.temporal.n_encoder_layers,
            n_decoder_layers=cfg.model.temporal.n_decoder_layers,
            dim_feedforward=cfg.model.temporal.dim_feedforward,
            dropout=cfg.model.temporal.dropout,
            forecast_steps=cfg.data.temporal.forecast_steps,
            beta_kl=cfg.model.temporal.beta_kl,
            free_bits=cfg.model.temporal.free_bits,
        )
        runner = TemporalStageRunner(
            model=temporal_model,
            encoders=encoders,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            output_dir=cfg.project.get("checkpoint_dir", "checkpoints"),
            domain_indices=domain_indices,
        )
        runner.run()

    elif stage == "decoder":
        from training.stage_runners import GeneratorStageRunner
        from models.spatial_encoder import DomainEncoderEnsemble
        from models.temporal_bayesian import VariationalTransformer
        from models.reverse_generator import ConditionalUNet

        domain_ch = _get_domain_channels(cfg)
        encoders = _load_encoders(cfg, domain_ch, device)
        meteo_dim = len(domain_indices.get("meteo", [])) or 1
        total_ch = _total_channels(cfg)

        temporal_model = _load_temporal(cfg, encoders, meteo_dim, device)
        generator = ConditionalUNet(
            state_dim=cfg.model.temporal.state_dim,
            out_channels=total_ch,
            noise_dim=cfg.model.generator.noise_dim,
            base_channels=cfg.model.generator.base_channels,
            channel_multipliers=list(cfg.model.generator.channel_multipliers),
            n_res_blocks=cfg.model.generator.n_res_blocks,
            use_spectral_norm=cfg.model.generator.use_spectral_norm,
            initial_spatial=tuple(cfg.model.generator.initial_spatial),
        )
        runner = GeneratorStageRunner(
            model=generator,
            encoders=encoders,
            temporal_model=temporal_model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            output_dir=cfg.project.get("checkpoint_dir", "checkpoints"),
            domain_indices=domain_indices,
        )
        runner.run()

    elif stage == "fusion":
        from training.stage_runners import FusionStageRunner
        from models.spatial_encoder import DomainEncoderEnsemble
        from models.temporal_bayesian import VariationalTransformer
        from models.reverse_generator import ConditionalUNet
        from models.fusion import FusionTransformer

        domain_ch = _get_domain_channels(cfg)
        encoders = _load_encoders(cfg, domain_ch, device)
        meteo_dim = len(domain_indices.get("meteo", [])) or 1
        total_ch = _total_channels(cfg)

        temporal_model = _load_temporal(cfg, encoders, meteo_dim, device)
        generator = _load_generator(cfg, total_ch, device)
        fusion = FusionTransformer(
            cnn_dim=encoders.combined_latent_dim,
            bayes_dim=cfg.model.temporal.state_dim * 2,
            meteo_dim=max(meteo_dim, 1),
            gen_dim=total_ch,
            d_model=cfg.model.fusion.d_model,
            n_heads=cfg.model.fusion.n_heads,
            n_layers=cfg.model.fusion.n_layers,
            dim_feedforward=cfg.model.fusion.dim_feedforward,
            dropout=cfg.model.fusion.dropout,
            out_channels=total_ch,
            spatial_size=(cfg.data.spatial.patch_size, cfg.data.spatial.patch_size),
        )
        runner = FusionStageRunner(
            model=fusion,
            encoders=encoders,
            temporal_model=temporal_model,
            generator=generator,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            output_dir=cfg.project.get("checkpoint_dir", "checkpoints"),
            domain_indices=domain_indices,
        )
        runner.run()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained model checkpoint on the test split."""
    # Delegate to scripts/evaluate.py logic
    from scripts.evaluate import run_evaluation

    cfg = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint).parent
    run_evaluation(cfg, checkpoint_dir)


def cmd_forecast(args: argparse.Namespace) -> None:
    """Run probabilistic forecast inference."""
    logger.info("Forecast command — not yet implemented")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_domain_channels(cfg) -> dict[str, int]:
    domain_ch: dict[str, int] = {}
    for product in cfg.data.products:
        d = product["domain"]
        if d == "meteo":
            continue
        domain_ch[d] = domain_ch.get(d, 0) + len(product["channels"])
    return domain_ch


def _total_channels(cfg) -> int:
    return sum(len(p["channels"]) for p in cfg.data.products)


def _build_domain_indices(cfg) -> dict[str, list[int]]:
    indices: dict[str, list[int]] = {}
    offset = 0
    for product in cfg.data.products:
        d = product["domain"]
        n = len(product["channels"])
        indices.setdefault(d, []).extend(range(offset, offset + n))
        offset += n
    return indices


def _load_encoders(cfg, domain_ch, device):
    """Load encoder ensemble from checkpoint."""
    import torch
    from models.spatial_encoder import DomainEncoderEnsemble

    encoders = DomainEncoderEnsemble(domain_ch, cfg.model.encoder)
    ckpt = Path(cfg.project.get("checkpoint_dir", "checkpoints")) / "encoders" / "best.pt"
    if ckpt.exists():
        encoders.load_state_dict(torch.load(ckpt, weights_only=True))
        logger.info(f"Loaded encoder checkpoint: {ckpt}")
    encoders.to(device)
    return encoders


def _load_temporal(cfg, encoders, meteo_dim, device):
    """Load temporal model from checkpoint."""
    import torch
    from models.temporal_bayesian import VariationalTransformer

    model = VariationalTransformer(
        latent_dim=encoders.combined_latent_dim,
        meteo_dim=max(meteo_dim, 1),
        state_dim=cfg.model.temporal.state_dim,
        d_model=cfg.model.temporal.d_model,
        n_heads=cfg.model.temporal.n_heads,
        n_encoder_layers=cfg.model.temporal.n_encoder_layers,
        n_decoder_layers=cfg.model.temporal.n_decoder_layers,
        dim_feedforward=cfg.model.temporal.dim_feedforward,
        dropout=cfg.model.temporal.dropout,
        forecast_steps=cfg.data.temporal.forecast_steps,
        beta_kl=cfg.model.temporal.beta_kl,
        free_bits=cfg.model.temporal.free_bits,
    )
    ckpt = Path(cfg.project.get("checkpoint_dir", "checkpoints")) / "temporal" / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        logger.info(f"Loaded temporal checkpoint: {ckpt}")
    model.to(device)
    return model


def _load_generator(cfg, total_ch, device):
    """Load generator from checkpoint."""
    import torch
    from models.reverse_generator import ConditionalUNet

    model = ConditionalUNet(
        state_dim=cfg.model.temporal.state_dim,
        out_channels=total_ch,
        noise_dim=cfg.model.generator.noise_dim,
        base_channels=cfg.model.generator.base_channels,
        channel_multipliers=list(cfg.model.generator.channel_multipliers),
        n_res_blocks=cfg.model.generator.n_res_blocks,
        use_spectral_norm=cfg.model.generator.use_spectral_norm,
        initial_spatial=tuple(cfg.model.generator.initial_spatial),
    )
    ckpt = Path(cfg.project.get("checkpoint_dir", "checkpoints")) / "generator" / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        logger.info(f"Loaded generator checkpoint: {ckpt}")
    model.to(device)
    return model


def _resolve_device(device_str: str) -> str:
    import torch
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        return "cpu"
    return device_str


def _set_seed(seed: int) -> None:
    from utils.reproducibility import set_global_seed
    set_global_seed(seed)
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

    # stream (download + preprocess in one pass, disk-efficient)
    p_stream = sub.add_parser("stream", help="Download + preprocess in streaming mode (disk-efficient)")
    p_stream.set_defaults(func=cmd_download_and_process)

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