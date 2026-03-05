#!/usr/bin/env python
"""Run the full preprocessing pipeline: raw NetCDF → Zarr store."""
import click
from pathlib import Path
from utils.config_loader import load_config
from utils.logger import get_logger
from data.preprocessor import GOESPreprocessor

logger = get_logger(__name__)


@click.command()
@click.option("--input", "input_dir", required=True, help="Input directory with raw NetCDF files")
@click.option("--output", required=True, help="Output Zarr store path")
@click.option("--config", "config_path", required=True, help="Path to config YAML")
def main(input_dir, output, config_path):
    cfg = load_config(config_path)
    logger.info(f"Preprocessing {input_dir} → {output}")

    preprocessor = GOESPreprocessor(cfg)
    input_path = Path(input_dir)

    # Process each product
    all_data = []
    channel_names = []
    for product in cfg.data.products:
        pid = product["id"]
        shortname = pid.split("-")[-1].replace("F", "")
        product_dir = input_path / shortname
        if not product_dir.exists():
            if product.get("required", True):
                logger.warning(f"Required product dir missing: {product_dir}")
            continue

        files = sorted(product_dir.glob("*.nc"))
        if not files:
            continue

        results = preprocessor.process_files(files, pid)
        for r in results:
            all_data.append(r["data"])
        channel_names.extend(product["channels"])

    if not all_data:
        logger.error("No data processed. Check input directory.")
        return

    import numpy as np
    # Stack along time dimension — simplified: treat each processed result as one timestep
    data = np.stack(all_data, axis=0)  # (T, C, H, W)
    times = np.arange(data.shape[0]).astype("datetime64[h]") + np.datetime64("2023-01-01")

    stats = preprocessor.compute_stats({
        name: data[:, i] for i, name in enumerate(channel_names) if i < data.shape[1]
    })
    preprocessor.write_zarr(output, data, channel_names, times, stats)
    logger.info("Preprocessing complete")


if __name__ == "__main__":
    main()
