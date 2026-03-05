#!/usr/bin/env python
"""Download GOES L2 products for a date range (disk-aware)."""
import click
from utils.logger import get_logger
from data.downloader import GOESDownloader

logger = get_logger(__name__)


@click.command()
@click.option("--satellite", default="goes16", help="Satellite name (e.g. goes16)")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--products", default="SST,LST,CMIP,DMW,TPW", help="Comma-separated product list")
@click.option("--output-dir", default="data/raw", help="Output directory")
@click.option("--interval", default=3, type=int, help="Hours between downloads (default 3)")
@click.option("--disk-budget", default=10.0, type=float, help="Max disk usage in GB (default 10)")
def main(satellite, start, end, products, output_dir, interval, disk_budget):
    product_list = [p.strip() for p in products.split(",")]
    logger.info(
        f"Downloading {product_list} from {satellite} ({start} to {end}), "
        f"interval={interval}h, budget={disk_budget}GB"
    )
    downloader = GOESDownloader(
        satellite=satellite,
        output_dir=output_dir,
        products=product_list,
        disk_budget_gb=disk_budget,
    )
    manifest = downloader.download(
        start=start,
        end=end,
        interval_hours=interval,
        cleanup_after_each=True,
    )
    logger.info(f"Download complete: {len(manifest)} files")


if __name__ == "__main__":
    main()