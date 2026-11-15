"""End-to-end analysis: data -> indicators -> rules + ML -> blended risk score (+ Pillar Two)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import benchmarks, metrics, model, pillar2, rules
from .config import Config, DEFAULT_CONFIG

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
    df = metrics.load_dataset(_read(data))
    ind = metrics.compute_indicators(df)
    X = metrics.feature_matrix(ind, config.thresholds.globe_minimum_rate)

    if models is None:
        models = model.train(training, config)

    r = rules.evaluate(ind, config.thresholds)
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
    return AnalysisResult(out, models, pillar2.assess(df, fiscal_year), fiscal_year, config)
