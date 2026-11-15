"""End-to-end analysis: data -> indicators -> rules + ML -> blended risk score (+ Pillar Two)."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import benchmarks, jurisdictions, metrics, model, pillar2, quality, rules
from .config import Config, DEFAULT_CONFIG

MONEY_COLUMNS = ["revenue", "profit_before_tax", "tax_paid", "tax_accrued", "tangible_assets", "payroll_costs"]

DISCLAIMER = (
    "TaxRiskScan produces risk indicators, not legal or tax conclusions. "
    "A high score means 'look closer', not 'non-compliant': low effective tax "
    "rates and high profits can have legitimate explanations (losses carried "
    "forward, incentives, timing differences, genuine IP returns). Every flag "
    "needs human review."
)


@dataclass
class AnalysisResult:
    results: pd.DataFrame
    models: model.RiskModels
    pillar2: pd.DataFrame
    fiscal_year: int
    config: Config = field(default=DEFAULT_CONFIG)
    issues: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        cols = ["group"] + (["year"] if "year" in self.results.columns else []) + [
            "jurisdiction", "risk_score", "risk_level", "etr", "oecd_etr_2016",
            "profit_per_employee", "profit_share", "activity_share",
            "rule_points", "ml_probability", "anomaly_score", "ml_drivers", "flags",
        ]
        out = self.results[cols].copy()
        out["flags"] = out["flags"].map(lambda f: ", ".join(f))
        return out.sort_values("risk_score", ascending=False).reset_index(drop=True)

    def level_counts(self) -> dict[str, int]:
        counts = self.results["risk_level"].value_counts()
        return {lvl: int(counts.get(lvl, 0)) for lvl in ("High", "Medium", "Low")}

    def to_dict(self) -> dict:
        """Everything a client needs, JSON-serialisable (used by the API and the web app)."""
        from .legal import basis_for

        r = self.results.sort_values("risk_score", ascending=False)
        entities = []
        for _, x in r.iterrows():
            entities.append({
                "group": x["group"],
                "year": _clean(x["year"]) if "year" in r.columns else None,
                "jurisdiction": x["jurisdiction"],
                "risk_score": _clean(x["risk_score"]),
                "risk_level": x["risk_level"],
                "etr": _clean(x["etr"]),
                "oecd_etr_2016": _clean(x["oecd_etr_2016"]),
                "profit_per_employee": _clean(x["profit_per_employee"]),
                "profit_share": _clean(x["profit_share"]),
                "activity_share": _clean(x["activity_share"]),
                "rule_points": _clean(x["rule_points"]),
                "ml_probability": _clean(x["ml_probability"]),
                "anomaly_score": _clean(x["anomaly_score"]),
                "ml_drivers": x["ml_drivers"],
                "findings": [
                    {"code": c, "reason": reason, "recommendation": rec,
                     "legal_basis": [{"label": s.label(), "url": s.url} for s in basis_for(c)]}
                    for c, reason, rec in zip(x["flags"], x["reasons"], x["recommendations"])
                ],
            })
        p2 = self.pillar2.sort_values("estimated_topup_tax", ascending=False)
        return {
            "metadata": self.metadata,
            "counts": self.level_counts(),
            "data_quality": [i.to_dict() for i in self.issues],
            "entities": entities,
            "pillar2": {
                "fiscal_year": self.fiscal_year,
                "jurisdictions": [{k: _clean(v) for k, v in row.items()} for row in p2.to_dict("records")],
                "groups": [{k: _clean(v) for k, v in row.items()}
                           for row in pillar2.group_totals(self.pillar2).to_dict("records")],
            },
            "model": self.models.card(),
            "disclaimer": DISCLAIMER,
        }


def _clean(v):
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _fingerprint(data) -> str:
    try:
        if isinstance(data, pd.DataFrame):
            return hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()
        if isinstance(data, (str, Path)):
            return hashlib.sha256(Path(data).read_bytes()).hexdigest()
        if hasattr(data, "getvalue"):
            raw = data.getvalue()
            return hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest()
    except OSError:
        pass
    return ""


def risk_level(score: float, config: Config) -> str:
    if score >= config.levels.high:
        return "High"
    if score >= config.levels.medium:
        return "Medium"
    return "Low"


def _read(data):
    if isinstance(data, (str, Path)) and str(data).lower().endswith(".xml"):
        from .cbcxml import read_cbc_xml
        return read_cbc_xml(data)
    if hasattr(data, "name") and str(getattr(data, "name", "")).lower().endswith(".xml"):
        from .cbcxml import read_cbc_xml
        return read_cbc_xml(data)
    return data


def _drivers(contrib: pd.DataFrame, top: int = 2, min_effect: float = 0.05) -> list[str]:
    out = []
    for _, row in contrib.iterrows():
        pos = row[row > min_effect].sort_values(ascending=False).head(top)
        out.append(", ".join(f"{model.FEATURE_LABELS[f]} (+{v:.2f})" for f, v in pos.items()))
    return out


def analyze(
    data,
    config: Config = DEFAULT_CONFIG,
    models: model.RiskModels | None = None,
    training=None,
    fiscal_year: int = 2026,
) -> AnalysisResult:
    """Analyse a CbCR-style dataset (CSV/XML path, file object or DataFrame)."""
    if hasattr(data, "read") and not hasattr(data, "getvalue"):
        name = getattr(data, "name", "")
        raw = data.read()
        data = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode())
        data.name = name
    digest = _fingerprint(data)
    if hasattr(data, "seek"):
        data.seek(0)
    df = metrics.load_dataset(_read(data))
    if config.eur_rate != 1.0:
        for col in MONEY_COLUMNS:
            if col in df.columns:
                df[col] = df[col] * config.eur_rate
    issues = quality.check(df)
    ind = metrics.compute_indicators(df)
    X = metrics.feature_matrix(ind, config.thresholds.globe_minimum_rate)

    if models is None:
        models = model.train(training, config)

    r = rules.evaluate(ind, config.thresholds, config.watchlist)
    out = pd.concat([ind, r, benchmarks.lookup(ind["jurisdiction"])], axis=1)
    out["ml_probability"] = models.probability(X)
    out["ml_drivers"] = _drivers(models.explain(X))
    out["anomaly_score"] = models.anomaly(X)
    out["cluster"] = models.cluster(X)

    w = config.weights
    total = w.rules + w.classifier + w.clustering
    out["risk_score"] = (
        100 * (w.rules * out["rule_score"] + w.classifier * out["ml_probability"]
               + w.clustering * out["anomaly_score"]) / total
    ).round(1)
    out["risk_level"] = out["risk_score"].map(lambda s: risk_level(s, config))
    from . import __version__

    metadata = {
        "tool": "TaxRiskScan",
        "version": __version__,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_sha256": digest,
        "rows": len(df),
        "groups": int(df["group"].nunique()),
        "currency": config.currency,
        "eur_rate": config.eur_rate,
        "organisation": config.organisation,
        "config_fingerprint": config.fingerprint(),
        "model_source": models.training_source,
        "model_data_sha256": models.data_sha256,
        "eu_annex_i_as_of": jurisdictions.EU_ANNEX_I_AS_OF,
        "fiscal_year_default": fiscal_year,
    }
    return AnalysisResult(out, models, pillar2.assess(df, fiscal_year), fiscal_year, config, issues, metadata)
