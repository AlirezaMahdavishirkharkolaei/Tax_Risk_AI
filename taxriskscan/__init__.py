"""TaxRiskScan - flag jurisdictions with tax-avoidance and profit-shifting risk."""

from .config import Config, DEFAULT_CONFIG
from .metrics import DataError, load_dataset
from .pipeline import DISCLAIMER, AnalysisResult, analyze

__all__ = ["Config", "DEFAULT_CONFIG", "DataError", "load_dataset", "analyze", "AnalysisResult", "DISCLAIMER"]
__version__ = "3.0.0"
