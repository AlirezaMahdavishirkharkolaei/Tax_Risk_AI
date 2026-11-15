"""Backward-compatible entry point.

The original single-file prototype has grown into the ``taxriskscan`` package.
Running this file analyses the bundled sample dataset and writes the report to
``results/``. For your own data use:

    python -m taxriskscan analyze path/to/data.csv --out results/
"""

import sys
from pathlib import Path

from taxriskscan.cli import main

if __name__ == "__main__":
    sample = Path(__file__).parent / "data" / "sample_cbcr.csv"
    sys.exit(main(["analyze", str(sample), "--out", "results"]))
