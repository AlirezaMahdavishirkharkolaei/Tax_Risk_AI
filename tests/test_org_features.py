import json
from pathlib import Path

import pandas as pd
import pytest

from taxriskscan import analyze, model, research
from taxriskscan.cli import main
from taxriskscan.config import ConfigError, config_from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_cbcr.csv"
XML = ROOT / "data" / "sample_cbc.xml"
EXAMPLE_CONFIG = ROOT / "taxriskscan" / "data" / "example_config.toml"


@pytest.fixture(scope="module")
def models():
    return model.train(cross_validate=False)


# --- configuration -------------------------------------------------------------------

def test_example_config_loads():
    c = load_config(EXAMPLE_CONFIG)
    assert c.currency == "USD" and c.eur_rate == 0.92 and "Panama" in c.watchlist
    assert len(c.fingerprint()) == 12 and c.fingerprint() != config_from_dict({}).fingerprint()


@pytest.mark.parametrize("bad, match", [
    ({"thresholds": {"nonsense": 1}}, "Unknown setting"),
    ({"levels": {"medium": 80, "high": 60}}, "Levels"),
    ({"weights": {"rules": -1}}, "non-negative"),
    ({"data": {"eur_rate": 0}}, "eur_rate"),
    ({"extra": {}}, "Unknown section"),
])
def test_invalid_config_is_rejected(bad, match):
    with pytest.raises(ConfigError, match=match):
        config_from_dict(bad)


def test_watchlist_flags_jurisdiction(models):
    res = analyze(SAMPLE, config=config_from_dict({"lists": {"watchlist": ["poland"]}}), models=models)
    poland = res.results.set_index("jurisdiction").loc["Poland"]
    assert "WATCHLIST" in poland["flags"]


def test_currency_conversion_changes_pillar2_thresholds(models):
    df = pd.DataFrame([{"jurisdiction": "Germany", "revenue": 9e6, "profit_before_tax": 9e5, "tax_paid": 0, "employees": 5}])
    eur = analyze(df, models=models).pillar2.iloc[0]
    assert eur["test_de_minimis"]
    # The same figures in a currency worth 2 EUR each are no longer de minimis.
    strong = analyze(df, models=models, config=config_from_dict({"data": {"currency": "XXX", "eur_rate": 2}})).pillar2.iloc[0]
    assert not strong["test_de_minimis"]


# --- data quality & audit trail ------------------------------------------------------------

def test_quality_checks(models):
    df = pd.read_csv(SAMPLE)
    df.loc[len(df)] = ["Kestrel Logistics", 2026, "Stateless", 1e6, 5e5, 0, 0, 0, 0]
    df.loc[len(df)] = ["Kestrel Logistics", 2026, "Germny", 1e6, 5e4, 6e4, 10, 0, 0]
    checks = {i.check for i in analyze(df, models=models).issues}
    assert {"aggregate_rows", "unknown_jurisdictions", "tax_exceeds_profit"} <= checks
    assert analyze(SAMPLE, models=models).issues == []


def test_metadata_and_json(models):
    res = analyze(SAMPLE, models=models)
    d = json.loads(json.dumps(res.to_dict()))
    meta = d["metadata"]
    assert len(meta["input_sha256"]) == 64 and meta["rows"] == 19 and meta["model_data_sha256"]
    assert d["entities"][0]["findings"][0]["legal_basis"]
    assert d["model"]["cv_roc_auc_groupwise"] is None  # trained without CV in this fixture
    assert "non-commercial" in d["model"]["data_terms"]


# --- saved models ----------------------------------------------------------------------------

def test_save_and_load_model(tmp_path, models):
    p = model.save(models, tmp_path / "m.joblib")
    loaded = model.load(p)
    a = analyze(SAMPLE, models=models).results["ml_probability"].values
    b = analyze(SAMPLE, models=loaded).results["ml_probability"].values
    assert (abs(a - b) < 1e-12).all()


def test_load_rejects_foreign_files(tmp_path):
    import joblib
    joblib.dump({"not": "ours"}, tmp_path / "x.joblib")
    with pytest.raises(Exception, match="not a TaxRiskScan model"):
        model.load(tmp_path / "x.joblib")


# --- CLI ----------------------------------------------------------------------------------------

def test_cli_batch_config_and_json(tmp_path, capsys):
    assert main(["analyze", str(SAMPLE), str(XML), "--out", str(tmp_path), "--config", str(EXAMPLE_CONFIG),
                 "--train", "synthetic", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out) == 2 and out[0]["metadata"]["currency"] == "USD"
    assert (tmp_path / "sample_cbcr" / "report.html").exists() and (tmp_path / "sample_cbc" / "result.json").exists()


def test_cli_train_save_and_reuse(tmp_path, capsys):
    m = tmp_path / "m.joblib"
    assert main(["train", "synthetic", "--save", str(m)]) == 0
    assert json.loads(Path(str(m) + ".card.json").read_text())["training_source"] == "training_cbcr.csv"
    assert main(["analyze", str(XML), "--model", str(m), "--out", str(tmp_path / "o")]) == 0
    assert "Bermuda" in capsys.readouterr().out


def test_cli_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("[levels]\nmedium = 90\nhigh = 10\n")
    assert main(["analyze", str(SAMPLE), "--config", str(bad), "--out", str(tmp_path)]) == 2
    assert "Levels" in capsys.readouterr().err


# --- research mode ------------------------------------------------------------------------------

def test_research_mode(tmp_path):
    r = research.run()
    assert r.overview["groups"] > 80 and 0.05 < r.overview["haven_share_profit"] < 0.25
    # Havens hold more profit than employees, and are taxed less, in every year.
    assert (r.by_year["profit_to_employee_ratio"].iloc[1:] > 1).all()
    assert (r.etr_by_year["haven_weighted_etr"] < r.etr_by_year["other_weighted_etr"]).all()
    assert 0.65 < r.validation["ml_auc_out_of_sample"] < 0.85
    assert r.validation["high_share_havens"] > r.validation["high_share_other"]
    paths = research.write_outputs(r, tmp_path)
    assert "Caveats" in paths["brief"].read_text() and (tmp_path / "top_havens.csv").exists()


def test_research_requires_labels():
    from taxriskscan import DataError
    with pytest.raises(DataError, match="label"):
        research.run(pd.read_csv(SAMPLE))


# --- REST API ----------------------------------------------------------------------------------

def test_api(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("TAXRISKSCAN_API_KEYS", "k1,k2")
    monkeypatch.setenv("TAXRISKSCAN_TRAINING", "synthetic")
    from taxriskscan.api import app

    with TestClient(app) as c:
        assert c.get("/health").json()["status"] == "ok"
        assert c.get("/model").status_code == 401
        h = {"X-API-Key": "k2"}
        r = c.post("/analyze", headers=h, files={"file": ("s.csv", SAMPLE.read_bytes(), "text/csv")})
        assert r.status_code == 200 and r.json()["counts"]["High"] >= 3
        r = c.post("/analyze", headers=h, files={"file": ("x.xml", XML.read_bytes())})
        assert r.status_code == 200 and len(r.json()["entities"]) == 4
        r = c.post("/analyze/json", headers=h, json={"entities": [
            {"jurisdiction": "Germany", "revenue": 1e7, "profit_before_tax": 1e6, "tax_paid": 3e5, "employees": 50},
            {"jurisdiction": "Bermuda", "revenue": 8e6, "profit_before_tax": 6e6, "tax_paid": 0, "employees": 2}]})
        assert r.json()["entities"][0]["jurisdiction"] == "Bermuda"
        assert c.post("/analyze", headers=h, files={"file": ("b.csv", b"a,b\n1,2\n")}).status_code == 422
        assert c.post("/report", headers=h, files={"file": ("s.csv", SAMPLE.read_bytes())}).text.startswith("<!DOCTYPE html>")
        assert c.get("/model", headers=h).json()["training_source"] == "training_cbcr.csv"
