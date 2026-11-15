"""Glue between the browser page (Pyodide) and the taxriskscan package.

Every function returns plain JSON-serialisable data so the page can render it.
The same code also runs under CPython, which is how the build script
pre-computes the sample result shown before Pyodide has loaded.
"""

from __future__ import annotations

import base64
import io
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from taxriskscan import analyze, model  # noqa: E402
from taxriskscan.legal import basis_for  # noqa: E402
from taxriskscan.pillar2 import group_totals  # noqa: E402
from taxriskscan.report import html_report, plot_heatmap, plot_scores  # noqa: E402

_MODELS: dict[str, model.RiskModels] = {}


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        return _clean(v.item())
    return v


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def get_models(training: str = "real") -> model.RiskModels:
    if training not in _MODELS:
        _MODELS[training] = model.train(training, cross_validate=False)
    return _MODELS[training]


def run(path: str, fiscal_year: int = 2026, training: str = "real", cv_auc: float | None = None) -> str:
    m = get_models(training)
    if cv_auc is not None:
        m.cv_auc = cv_auc
    res = analyze(path, models=m, fiscal_year=fiscal_year)
    r = res.results.sort_values("risk_score", ascending=False)

    entities = []
    for _, x in r.iterrows():
        entities.append({
            "group": x["group"],
            "year": _clean(x["year"]) if "year" in r.columns else None,
            "jurisdiction": x["jurisdiction"],
            "score": _clean(x["risk_score"]),
            "level": x["risk_level"],
            "etr": _clean(x["etr"]),
            "oecd_etr": _clean(x["oecd_etr_2016"]),
            "ppe": _clean(x["profit_per_employee"]),
            "profit_share": _clean(x["profit_share"]),
            "activity_share": _clean(x["activity_share"]),
            "ml": _clean(x["ml_probability"]),
            "anomaly": _clean(x["anomaly_score"]),
            "drivers": x["ml_drivers"],
            "findings": [
                {
                    "code": code, "reason": reason, "recommendation": rec,
                    "law": [{"label": s.label(), "url": s.url} for s in basis_for(code)],
                }
                for code, reason, rec in zip(x["flags"], x["reasons"], x["recommendations"])
            ],
        })

    p2 = res.pillar2.sort_values("estimated_topup_tax", ascending=False)
    pillar = [{k: _clean(v) for k, v in row.items()} for row in p2.to_dict("records")]
    totals = [{k: _clean(v) for k, v in row.items()} for row in group_totals(res.pillar2).to_dict("records")]

    out = {
        "counts": res.level_counts(),
        "entities": entities,
        "pillar2": pillar,
        "pillar2_totals": totals,
        "fiscal_year": res.fiscal_year,
        "model": {
            "source": m.training_source, "rows": m.n_training, "positive": m.n_positive,
            "cv_auc": _clean(m.cv_auc), "training": training,
        },
        "heatmap": _png(plot_heatmap(res)),
        "scores": _png(plot_scores(res)),
        "report_html": html_report(res),
        "summary_csv": res.summary().to_csv(index=False),
        "pillar2_csv": res.pillar2.to_csv(index=False),
        "groups": int(r["group"].nunique()),
        "issues": [i.to_dict() for i in res.issues],
        "metadata": res.metadata,
    }
    return json.dumps(out)


def template_csv() -> str:
    from taxriskscan.cli import TEMPLATE
    return TEMPLATE


def peek(path: str) -> str:
    """Row count and columns of an uploaded CSV, for the page's status line."""
    df = pd.read_csv(path, nrows=5)
    return json.dumps({"columns": list(df.columns)})
