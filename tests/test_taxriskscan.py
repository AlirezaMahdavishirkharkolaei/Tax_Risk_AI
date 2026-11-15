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
    return model.train()  # real public CbCR data


@pytest.fixture(scope="module")
def synthetic_models():
    return model.train("synthetic")


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

def test_synthetic_classifier_generalises(synthetic_models):
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from generate_synthetic_data import simulate

    assert synthetic_models.cv_auc > 0.9
    assert model.evaluate_on(synthetic_models, simulate(40, seed=123)) > 0.85


def test_real_training_data_and_model(models):
    df = pd.read_csv(model.default_training_path())
    assert len(df) > 4000 and df["group"].nunique() > 80
    assert 0.1 < df["label"].mean() < 0.25
    # Honest, group-wise cross-validated performance on real data (label noise is substantial).
    assert 0.68 < models.cv_auc < 0.85
    assert models.training_source == "training_public_cbcr.csv"


def test_jurisdiction_lists_are_not_model_inputs():
    from taxriskscan.metrics import FEATURES
    assert not any("cit" in f or "annex" in f or "cooperative" in f for f in FEATURES)


def test_explanations_point_to_low_tax_for_a_shell(models):
    ind = frame([("Germany", 5e8, 2e7, 5e6, 2000), ("Somewhere", 9e7, 6e7, 0, 3)])
    from taxriskscan.metrics import feature_matrix
    contrib = models.explain(feature_matrix(ind)).loc[1]
    assert contrib.sum() > 0.3 and contrib["low_tax_gap"] > 0


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
    for name in ("risk_summary.csv", "risk_heatmap.png", "risk_scores.png", "pillar2_summary.csv", "report.html"):
        assert (tmp_path / name).stat().st_size > 0
    html = (tmp_path / "report.html").read_text()
    assert "Findings" in html and "Cayman Islands" in html
    assert "Pillar Two screening" in html and "eur-lex.europa.eu" in html
    assert "High:" in capsys.readouterr().out


def test_cli_reports_bad_input(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    assert main(["analyze", str(bad), "--out", str(tmp_path)]) == 2
    assert "Missing required column" in capsys.readouterr().err


# --- Pillar Two -----------------------------------------------------------------

from taxriskscan import pillar2  # noqa: E402


def _p2(rows, year=2026):
    cols = ["jurisdiction", "revenue", "profit_before_tax", "tax_paid", "employees", "tangible_assets", "payroll_costs"]
    df = pd.DataFrame(rows, columns=cols)
    df["group"], df["year"] = "G", year
    return pillar2.assess(load_dataset(df)).set_index("jurisdiction")


def test_transition_and_sbie_rates():
    assert pillar2.transition_rate(2024) == 0.15
    assert pillar2.transition_rate(2025) == 0.16
    assert pillar2.transition_rate(2026) == pillar2.transition_rate(2027) == 0.17
    assert pillar2.transition_rate(2028) is None
    assert pillar2.sbie_rates(2023) == (0.098, 0.078)
    assert pillar2.sbie_rates(2026) == (0.092, 0.072)
    assert pillar2.sbie_rates(2040) == (0.05, 0.05)


def test_safe_harbour_tests():
    r = _p2([
        ("Small", 9e6, 9e5, 0, 5, 0, 0),                 # de minimis
        ("HighTax", 1e9, 1e8, 2.5e7, 900, 1e8, 6e7),     # simplified ETR 25% >= 17%
        ("Loss", 1e8, -5e6, 0, 100, 1e7, 5e6),           # routine profits (loss)
        ("Hub", 2e8, 1e8, 2e6, 4, 1e5, 4e5),             # fails all
    ])
    assert r.loc["Small", "test_de_minimis"] and r.loc["Small", "safe_harbour"] == "Applies"
    assert r.loc["HighTax", "test_simplified_etr"] and r.loc["HighTax", "estimated_topup_tax"] == 0
    assert r.loc["Loss", "test_routine_profits"] and r.loc["Loss", "safe_harbour"] == "Applies"
    hub = r.loc["Hub"]
    assert hub["safe_harbour"] == "Fails"
    sbie = 0.092 * 4e5 + 0.072 * 1e5
    assert hub["estimated_topup_tax"] == pytest.approx((0.15 - 0.02) * (1e8 - sbie))


def test_missing_payroll_gives_undetermined_upper_bound():
    df = pd.DataFrame([("G", "Hub", 2e8, 1e8, 2e6, 4, 1e5)],
                      columns=["group", "jurisdiction", "revenue", "profit_before_tax", "tax_paid", "employees", "tangible_assets"])
    r = pillar2.assess(load_dataset(df), fiscal_year=2026).iloc[0]
    assert r["test_routine_profits"] is None and r["safe_harbour"] == "Undetermined"
    assert r["estimate_is_upper_bound"]


def test_jurisdictional_blending():
    r = _p2([("X", 1e8, 1e7, 0, 10, 0, 0), ("X", 1e8, 1e7, 4e6, 10, 0, 0)])
    assert r.loc["X", "entities"] == 2 and r.loc["X", "simplified_etr"] == pytest.approx(0.2)


# --- legal basis, benchmarks, XML ----------------------------------------------------

def test_every_rule_has_a_legal_basis():
    from taxriskscan import legal, rules
    import inspect
    codes = set(__import__("re").findall(r'"([A-Z_]{5,})", \d', inspect.getsource(rules)))
    assert codes and all(legal.basis_for(c) for c in codes)
    assert all(s.url.startswith("https://") for c in codes for s in legal.basis_for(c) if s.url)


def test_oecd_benchmarks():
    from taxriskscan import benchmarks
    b = benchmarks.lookup(pd.Series(["Germany", "Bermuda", "Korea", "Atlantis"]))
    assert 0.15 < b.loc[0, "oecd_etr_2016"] < 0.30
    assert b.loc[1, "oecd_etr_2016"] < 0.05 and b.loc[1, "oecd_profit_employee_ratio_2016"] > 50
    assert not math.isnan(b.loc[2, "oecd_etr_2016"]) and math.isnan(b.loc[3, "oecd_etr_2016"])


def test_cbc_xml_import(models):
    from taxriskscan.cbcxml import read_cbc_xml
    df = read_cbc_xml(ROOT / "data" / "sample_cbc.xml")
    assert set(df["jurisdiction"]) == {"Germany", "Ireland", "Bermuda", "Poland"}
    assert df["year"].iloc[0] == 2026 and df["group"].iloc[0] == "Nordwind Electronics"
    res = analyze(ROOT / "data" / "sample_cbc.xml", models=models)
    assert res.results.set_index("jurisdiction").loc["Bermuda", "risk_level"] == "High"


def test_cbc_xml_rejects_entities(tmp_path):
    from taxriskscan.cbcxml import read_cbc_xml
    bad = tmp_path / "x.xml"
    bad.write_text('<?xml version="1.0"?><!DOCTYPE a [<!ENTITY x "y">]><a>&x;</a>')
    with pytest.raises(DataError):
        read_cbc_xml(bad)
