#!/usr/bin/env python
"""Download GOES L2 products for a date range."""
import click
from goes_forecast.utils.logger import get_logger
from goes_forecast.data.downloader import GOESDownloader

logger = get_logger(__name__)


@click.command()
@click.option("--satellite", default="GOES-16", help="Satellite name")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--products", default="SST,LST,CMIP,DMW,TPW", help="Comma-separated product list")
@click.option("--output-dir", default="data/raw", help="Output directory")
def main(satellite, start, end, products, output_dir):
    product_list = [p.strip() for p in products.split(",")]
    logger.info(f"Downloading {product_list} from {satellite} ({start} to {end})")
    downloader = GOESDownloader(satellite=satellite, output_dir=output_dir, products=product_list)
    manifest = downloader.download(start=start, end=end)
    logger.info(f"Download complete: {len(manifest)} files")


if __name__ == "__main__":
    main()
