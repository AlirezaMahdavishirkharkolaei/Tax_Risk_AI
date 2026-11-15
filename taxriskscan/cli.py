"""Command-line interface.

    taxriskscan analyze data/sample_cbcr.csv --out results/
    taxriskscan analyze my_cbc_report.xml          # OECD CbC XML (schema v2.0)
    taxriskscan train                              # diagnostics for the bundled real-data model
    taxriskscan train my_labelled_data.csv         # ... or for your own labelled cases
    taxriskscan sample > my_template.csv           # print a CSV template
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
    "group,year,jurisdiction,revenue,profit_before_tax,tax_paid,tax_accrued,employees,tangible_assets,payroll_costs\n"
    "Example Group,2026,Germany,10000000,2000000,450000,480000,50,3000000,3600000\n"
    "Example Group,2026,Ireland,5000000,1000000,120000,125000,20,400000,2800000\n"
    "Example Group,2026,Bermuda,8000000,3000000,0,0,5,0,900000\n"
)


def _cmd_analyze(a) -> int:
    from .report import write_outputs  # imports matplotlib only when needed

    res = analyze(a.csv, training=a.train, fiscal_year=a.fiscal_year)
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
    p2 = res.pillar2
    print(f"Pillar Two: safe harbour applies in {int((p2['safe_harbour'] == 'Applies').sum())}/{len(p2)} "
          f"jurisdictions, estimated top-up tax {p2['estimated_topup_tax'].sum():,.0f}")
    for name, p in paths.items():
        print(f"  {name:<8} {p}")
    print(f"\n{DISCLAIMER}")
    return 0


def _cmd_train(a) -> int:
    from . import metrics

    m = model.train(a.csv)
    auc = "n/a (needs >= 5 groups and >= 5 rows per class)" if m.cv_auc is None else f"{m.cv_auc:.3f}"
    print(f"Training data: {m.training_source} - {m.n_training:,} rows, {m.n_positive:,} positive")
    print(f"Group-wise cross-validated ROC AUC: {auc}")
    df = metrics.load_dataset(model.resolve_training(a.csv))
    X = metrics.feature_matrix(metrics.compute_indicators(df))
    print("Mean absolute effect of each feature on the probability:")
    imp = m.importance(X).rename(index=model.FEATURE_LABELS)
    print(imp.round(3).to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="taxriskscan", description="Flag high-risk tax jurisdictions in CbCR-style data.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="score a dataset and write the report")
    pa.add_argument("csv", type=Path, help=f"CSV with columns: {', '.join(REQUIRED_COLUMNS)} (optional: {', '.join(OPTIONAL_COLUMNS)}), or an OECD CbC XML file")
    pa.add_argument("--out", type=Path, default=Path("results"), help="output folder (default: results/)")
    pa.add_argument("--train", default=None, help="'real' (default, public CbCR data), 'synthetic', or a labelled CSV")
    pa.add_argument("--fiscal-year", type=int, default=2026, help="fiscal year for Pillar Two tests when the data has no 'year' column")
    pa.add_argument("--title", default="TaxRiskScan report")
    pa.set_defaults(func=_cmd_analyze)

    pt = sub.add_parser("train", help="fit the models on a labelled CSV and print diagnostics")
    pt.add_argument("csv", nargs="?", default=None, help="'real' (default), 'synthetic' or a labelled CSV")
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
