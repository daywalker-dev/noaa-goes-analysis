from setuptools import setup, find_packages

setup(
    name="goes_forecast",
    version="0.1.0",
    description="Probabilistic Earth-system forecasting from GOES L2 satellite data",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2", "numpy>=1.24", "scipy>=1.11",
        "omegaconf>=2.3", "pydantic>=2.0", "rich>=13.0", "click>=8.1",
    ],
    extras_require={
        "satellite": [
            "xarray>=2023.1", "zarr>=2.16", "numcodecs>=0.12",
            "goes2go>=2022.7.15", "pyresample>=1.27", "pyproj>=3.6",
        ],
        "metrics": ["pytorch-msssim>=1.0", "properscoring>=0.1"],
        "viz": ["matplotlib>=3.7", "cartopy>=0.21", "cmocean>=3.0"],
        "tracking": ["wandb>=0.15"],
        "dev": ["pytest>=7.4", "pytest-cov>=4.1", "ruff>=0.1", "mypy>=1.5"],
    },
)
