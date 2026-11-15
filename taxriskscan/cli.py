"""Command-line interface.

    taxriskscan analyze data.csv --out results/              # one file
    taxriskscan analyze reports/*.csv reports/*.xml --out results/   # batch: one sub-folder per file
    taxriskscan analyze data.csv --config org.toml --model my_model.joblib --json
    taxriskscan train my_cases.csv --save my_model.joblib    # train on your own labelled cases
    taxriskscan research --out research/                     # dataset-level research brief
    taxriskscan serve --port 8000                            # REST API (pip install "taxriskscan[api]")
    taxriskscan sample > template.csv                        # CSV template
    taxriskscan config > org.toml                            # example configuration

The default model can be chosen with the TAXRISKSCAN_TRAINING environment
variable: 'real' (default, non-commercial data terms), 'synthetic', or a path
to a labelled CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from . import __version__, model
from .config import DEFAULT_CONFIG, ConfigError, load_config
from .metrics import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, DataError
from .pipeline import DISCLAIMER, analyze

TEMPLATE = (
    "group,year,jurisdiction,revenue,profit_before_tax,tax_paid,tax_accrued,employees,tangible_assets,payroll_costs\n"
    "Example Group,2026,Germany,10000000,2000000,450000,480000,50,3000000,3600000\n"
    "Example Group,2026,Ireland,5000000,1000000,120000,125000,20,400000,2800000\n"
    "Example Group,2026,Bermuda,8000000,3000000,0,0,5,0,900000\n"
)


def default_training() -> str:
    return os.environ.get("TAXRISKSCAN_TRAINING", "real")


def _models(a, config):
    if getattr(a, "model", None):
        return model.load(a.model)
    return model.train(a.train or default_training(), config)


def _print_result(res, quiet_header: str = "") -> None:
    s = res.summary()
    view = s[["group", "jurisdiction", "risk_score", "risk_level", "etr", "flags"]].copy()
    view["etr"] = view["etr"].map(lambda x: "–" if pd.isna(x) else f"{x:.1%}")
    if view["group"].nunique() == 1:
        view = view.drop(columns="group")
    if quiet_header:
        print(quiet_header)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200):
        print(view.to_string(index=False))
    c = res.level_counts()
    print(f"\nHigh: {c['High']}  Medium: {c['Medium']}  Low: {c['Low']}")
    p2 = res.pillar2
    print(f"Pillar Two: safe harbour applies in {int((p2['safe_harbour'] == 'Applies').sum())}/{len(p2)} "
          f"jurisdictions, estimated top-up tax {p2['estimated_topup_tax'].sum():,.0f} EUR")
    for issue in res.issues:
        if issue.level == "warning":
            print(f"data warning: {issue.message}")


def _cmd_analyze(a) -> int:
    from .report import write_outputs  # imports matplotlib only when needed

    config = load_config(a.config) if a.config else DEFAULT_CONFIG
    models = _models(a, config)
    batch = len(a.inputs) > 1
    payload = []
    for path in a.inputs:
        res = analyze(path, config=config, models=models, fiscal_year=a.fiscal_year)
        out = a.out / Path(path).stem if batch else a.out
        paths = write_outputs(res, out, title=a.title)
        d = res.to_dict()
        (out / "result.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        if a.json:
            payload.append({"input": str(path), **d})
            continue
        _print_result(res, f"\n== {path}" if batch else "")
        for name, p in {**paths, "json": out / "result.json"}.items():
            print(f"  {name:<8} {p}")
    if a.json:
        print(json.dumps(payload if batch else payload[0], indent=1))
    else:
        print(f"\n{DISCLAIMER}")
    return 0


def _cmd_train(a) -> int:
    from . import metrics

    source = a.csv or default_training()
    m = model.train(source)
    auc = "n/a (needs >= 5 groups and >= 5 rows per class)" if m.cv_auc is None else f"{m.cv_auc:.3f}"
    print(f"Training data: {m.training_source} - {m.n_training:,} rows, {m.n_positive:,} positive")
    print(f"Group-wise cross-validated ROC AUC: {auc}")
    print(f"Data terms: {m.data_terms}")
    df = metrics.load_dataset(model.resolve_training(source))
    X = metrics.feature_matrix(metrics.compute_indicators(df))
    print("Mean absolute effect of each feature on the probability:")
    print(m.importance(X).rename(index=model.FEATURE_LABELS).round(3).to_string())
    if a.save:
        model.save(m, a.save)
        Path(str(a.save) + ".card.json").write_text(json.dumps(m.card(), indent=1), encoding="utf-8")
        print(f"Saved model to {a.save} (model card: {a.save}.card.json)")
    return 0


def _cmd_research(a) -> int:
    from . import research

    config = load_config(a.config) if a.config else DEFAULT_CONFIG
    res = research.run(a.csv, config)
    paths = research.write_outputs(res, a.out)
    o, v = res.overview, res.validation
    print(f"{o['groups']} multinationals, {o['reports']} reports ({o['years']}): "
          f"{o['haven_share_profit']:.1%} of profit and {o['haven_share_employees']:.1%} of employees in tax havens")
    print(f"Out-of-sample ROC AUC: classifier {v['ml_auc_out_of_sample']:.3f}, blended score {v['score_auc_out_of_sample']:.3f}")
    for name, p in paths.items():
        print(f"  {name:<24} {p}")
    return 0


def _cmd_serve(a) -> int:
    try:
        import uvicorn
    except ImportError:
        print('error: the API needs extra packages: pip install "taxriskscan[api]"', file=sys.stderr)
        return 2
    uvicorn.run("taxriskscan.api:app", host=a.host, port=a.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="taxriskscan",
                                description="Tax-avoidance, profit-shifting and Pillar Two screening for CbCR data.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pa = sub.add_parser("analyze", help="score one or more datasets and write reports")
    pa.add_argument("inputs", type=Path, nargs="+",
                    help=f"CSV ({', '.join(REQUIRED_COLUMNS)}; optional {', '.join(OPTIONAL_COLUMNS)}) or OECD CbC XML")
    pa.add_argument("--out", type=Path, default=Path("results"), help="output folder (default: results/)")
    pa.add_argument("--config", type=Path, help="TOML configuration (see 'taxriskscan config')")
    pa.add_argument("--model", type=Path, help="saved model from 'taxriskscan train --save'")
    pa.add_argument("--train", default=None, help="'real', 'synthetic' or a labelled CSV (ignored with --model)")
    pa.add_argument("--fiscal-year", type=int, default=2026, help="fiscal year for Pillar Two when the data has no 'year'")
    pa.add_argument("--title", default="TaxRiskScan report")
    pa.add_argument("--json", action="store_true", help="print the full result as JSON instead of a table")
    pa.set_defaults(func=_cmd_analyze)

    pt = sub.add_parser("train", help="fit the models, print diagnostics, optionally save them")
    pt.add_argument("csv", nargs="?", default=None, help="'real', 'synthetic' or a labelled CSV")
    pt.add_argument("--save", type=Path, help="write the trained model to this file (.joblib)")
    pt.set_defaults(func=_cmd_train)

    pr = sub.add_parser("research", help="dataset-level research brief (default: bundled public CbCR data)")
    pr.add_argument("csv", nargs="?", default=None, help="CSV with a 'label' column marking tax havens")
    pr.add_argument("--out", type=Path, default=Path("research"))
    pr.add_argument("--config", type=Path)
    pr.set_defaults(func=_cmd_research)

    pv = sub.add_parser("serve", help="run the REST API")
    pv.add_argument("--host", default="127.0.0.1")
    pv.add_argument("--port", type=int, default=8000)
    pv.set_defaults(func=_cmd_serve)

    ps = sub.add_parser("sample", help="print a CSV template")
    ps.set_defaults(func=lambda a: (sys.stdout.write(TEMPLATE), 0)[1])

    pc = sub.add_parser("config", help="print an example TOML configuration")
    pc.set_defaults(func=lambda a: (sys.stdout.write(_example_config()), 0)[1])

    a = p.parse_args(argv)
    try:
        return a.func(a)
    except (DataError, ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _example_config() -> str:
    from importlib import resources
    return (resources.files("taxriskscan") / "data" / "example_config.toml").read_text(encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
