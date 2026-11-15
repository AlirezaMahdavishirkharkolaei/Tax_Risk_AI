"""Command-line interface.

    taxriskscan analyze data/sample_cbcr.csv --out results/
    taxriskscan train my_labelled_data.csv        # check how well the models fit your labels
    taxriskscan sample > my_template.csv          # print a CSV template
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__, model
from .metrics import DataError, REQUIRED_COLUMNS, OPTIONAL_COLUMNS
from .pipeline import DISCLAIMER, analyze

TEMPLATE = (
    "group,jurisdiction,revenue,profit_before_tax,tax_paid,employees,tangible_assets\n"
    "Example Group,Germany,10000000,2000000,450000,50,3000000\n"
    "Example Group,Ireland,5000000,1000000,120000,20,400000\n"
    "Example Group,Bermuda,8000000,3000000,0,5,0\n"
)


def _cmd_analyze(a) -> int:
    from .report import write_outputs  # imports matplotlib only when needed

    res = analyze(a.csv, training=a.train)
    paths = write_outputs(res, a.out, title=a.title)
    s = res.summary()
    view = s[["group", "jurisdiction", "risk_score", "risk_level", "etr", "flags"]].copy()
    view["etr"] = view["etr"].map(lambda x: "–" if pd.isna(x) else f"{x:.1%}")
    if view["group"].nunique() == 1:
        view = view.drop(columns="group")
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(view.to_string(index=False))
    c = res.level_counts()
    print(f"\nHigh: {c['High']}  Medium: {c['Medium']}  Low: {c['Low']}")
    for name, p in paths.items():
        print(f"  {name:<8} {p}")
    print(f"\n{DISCLAIMER}")
    return 0


def _cmd_train(a) -> int:
    m = model.train(a.csv)
    auc = "n/a (needs >= 5 groups and >= 5 rows per class)" if m.cv_auc is None else f"{m.cv_auc:.3f}"
    print(f"Group-wise cross-validated ROC AUC: {auc}")
    print("Logistic-regression coefficients (standardised features):")
    print(m.coefficients().round(3).to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="taxriskscan", description="Flag high-risk tax jurisdictions in CbCR-style data.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="score a dataset and write the report")
    pa.add_argument("csv", type=Path, help=f"CSV with columns: {', '.join(REQUIRED_COLUMNS)} (optional: {', '.join(OPTIONAL_COLUMNS)})")
    pa.add_argument("--out", type=Path, default=Path("results"), help="output folder (default: results/)")
    pa.add_argument("--train", type=Path, default=None, help="labelled CSV to train the models on instead of the bundled data")
    pa.add_argument("--title", default="TaxRiskScan report")
    pa.set_defaults(func=_cmd_analyze)

    pt = sub.add_parser("train", help="fit the models on a labelled CSV and print diagnostics")
    pt.add_argument("csv", type=Path, nargs="?", default=None, help="labelled CSV (default: bundled synthetic data)")
    pt.set_defaults(func=_cmd_train)

    ps = sub.add_parser("sample", help="print a CSV template")
    ps.set_defaults(func=lambda a: (sys.stdout.write(TEMPLATE), 0)[1])

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except (DataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
