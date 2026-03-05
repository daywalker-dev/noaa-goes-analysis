"""GOES L2 Probabilistic Earth-System Forecasting Framework."""
__version__ = "0.1.0"

# Note: 'config' is a data directory (YAML files), not a Python package.
# Only import actual Python packages here.
import utils
import training
import models
import evaluation
import data