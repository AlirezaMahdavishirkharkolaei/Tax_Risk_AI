import math
from pathlib import Path

import pandas as pd
import pytest

from taxriskscan import DataError, analyze, jurisdictions, load_dataset, model
from taxriskscan.cli import main
from taxriskscan.config import Thresholds
from taxriskscan.metrics import compute_indicators
from taxriskscan.rules import evaluate_row

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_cbcr.csv"


@pytest.fixture(scope="module")
def models():
    return model.train()


def frame(rows, group="G"):
    df = pd.DataFrame(rows, columns=["jurisdiction", "revenue", "profit_before_tax", "tax_paid", "employees"])
    df["group"] = group
    return compute_indicators(load_dataset(df))


# --- data loading ----------------------------------------------------------

def test_column_aliases_are_accepted():
    df = pd.DataFrame({"Country": ["Germany"], "Turnover": [100], "Profit": [10], "Tax": [3], "Emps": [2]})
    out = load_dataset(df)
    assert list(out.columns[:5]) == ["jurisdiction", "revenue", "profit_before_tax", "tax_paid", "employees"]


def test_missing_columns_raise():
    with pytest.raises(DataError, match="employees"):
        load_dataset(pd.DataFrame({"jurisdiction": ["X"], "revenue": [1], "profit_before_tax": [1], "tax_paid": [0]}))


def test_non_numeric_values_raise():
    df = pd.DataFrame({"jurisdiction": ["X"], "revenue": ["abc"], "profit_before_tax": [1], "tax_paid": [0], "employees": [1]})
    with pytest.raises(DataError, match="Non-numeric"):
        load_dataset(df)


# --- indicators (regression tests for the original prototype's bugs) -------

def test_loss_making_entity_has_no_etr_and_is_not_flagged():
    ind = frame([("Germany", 1e6, -5e5, 0, 10), ("France", 1e6, 1e5, 25e3, 10)])
    assert math.isnan(ind.loc[0, "etr"])
    assert evaluate_row(ind.loc[0], Thresholds()) == []


def test_loss_making_entity_paying_tax_has_no_absurd_etr():
    ind = frame([("Germany", 1e6, -5e5, 2e4, 10)])
    assert math.isnan(ind.loc[0, "etr"])


def test_etr_and_profit_activity_gap():
    ind = frame([("A", 100, 50, 5, 1), ("B", 100, 50, 15, 9)])
    assert ind.loc[0, "etr"] == pytest.approx(0.10)
    assert ind.loc[0, "profit_share"] == pytest.approx(0.5)
    assert ind.loc[0, "activity_share"] == pytest.approx(0.1)
    assert ind.loc[0, "profit_activity_gap"] == pytest.approx(0.4)


# --- rules ------------------------------------------------------------------

def test_rules_fire_for_shell_in_listed_jurisdiction():
    ind = frame([("Anguilla", 1e7, 8e6, 0, 0), ("Germany", 5e7, 2e6, 6e5, 300)])
    codes = {f.code for f in evaluate_row(ind.loc[0], Thresholds())}
    assert {"ETR_VERY_LOW", "NO_EMPLOYEES", "PROFIT_SUBSTANCE_STRONG", "EU_ANNEX_I", "ZERO_CIT"} <= codes


def test_globe_band():
    ind = frame([("X", 1e6, 1e5, 12e3, 50)])
    codes = {f.code for f in evaluate_row(ind.loc[0], Thresholds())}
    assert "ETR_BELOW_GLOBE" in codes and "ETR_VERY_LOW" not in codes


# --- jurisdictions ------------------------------------------------------------

@pytest.mark.parametrize("name", ["russia", "  RUSSIAN federation ", "Turks & Caicos", "usvi"])
def test_aliases_normalise_to_annex_i(name):
    assert jurisdictions.is_eu_non_cooperative(name)


def test_panama_removed_in_october_2026():
    assert not jurisdictions.is_eu_non_cooperative("Panama")


# --- models -------------------------------------------------------------------

def test_classifier_quality_on_fresh_synthetic_data(models):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_synthetic_data import simulate

    assert models.cv_auc > 0.9
    assert model.evaluate_on(models, simulate(40, seed=123)) > 0.85


def test_typical_entity_is_not_anomalous(models):
    res = analyze(SAMPLE, models=models)
    germany = res.results[res.results["jurisdiction"] == "Germany"].iloc[0]
    assert germany["anomaly_score"] < 0.2


def test_training_requires_labels():
    df = pd.read_csv(SAMPLE)
    with pytest.raises(DataError, match="label"):
        model.train(df)


# --- end to end ---------------------------------------------------------------

def test_sample_ranking(models):
    res = analyze(SAMPLE, models=models)
    s = res.summary()
    high = set(s.loc[s["risk_level"] == "High", "jurisdiction"])
    assert {"Cayman Islands", "Bermuda", "Turks and Caicos Islands"} <= high
    low = set(s.loc[s["risk_level"] == "Low", "jurisdiction"])
    assert {"Germany", "France", "Turkey", "Mexico"} <= low
    assert s["risk_score"].between(0, 100).all()


def test_cli_writes_report(tmp_path, capsys):
    assert main(["analyze", str(SAMPLE), "--out", str(tmp_path)]) == 0
    for name in ("risk_summary.csv", "risk_heatmap.png", "risk_scores.png", "report.html"):
        assert (tmp_path / name).stat().st_size > 0
    html = (tmp_path / "report.html").read_text()
    assert "Findings" in html and "Cayman Islands" in html
    assert "High:" in capsys.readouterr().out


def test_cli_reports_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    assert main(["analyze", str(bad), "--out", str(tmp_path)]) == 2
    assert "Missing required column" in capsys.readouterr().err
