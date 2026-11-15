"""Using TaxRiskScan from Python (research notebooks, pipelines).

    python examples/python_api.py
"""

from pathlib import Path

import pandas as pd

from taxriskscan import analyze, model, research
from taxriskscan.config import config_from_dict
from taxriskscan.report import write_outputs

ROOT = Path(__file__).resolve().parents[1]

# 1. Score a dataset with the default (real-data) model.
res = analyze(ROOT / "data" / "sample_cbcr.csv")
print(res.summary()[["group", "jurisdiction", "risk_score", "risk_level"]].head())

# 2. Organisation settings in code (same keys as the TOML file).
config = config_from_dict({
    "organisation": {"name": "Research Group on International Tax"},
    "levels": {"medium": 30, "high": 55},
    "lists": {"watchlist": ["Cyprus"]},
})
models = model.train(cross_validate=False)          # train once, reuse for many files
res = analyze(ROOT / "data" / "sample_cbcr.csv", config=config, models=models)
write_outputs(res, "results/example")              # HTML report, CSVs, charts

# 3. A DataFrame works as input too.
df = pd.DataFrame([
    {"jurisdiction": "Germany", "revenue": 5e8, "profit_before_tax": 2e7, "tax_paid": 6e6, "employees": 2000},
    {"jurisdiction": "Luxembourg", "revenue": 9e7, "profit_before_tax": 6e7, "tax_paid": 1e6, "employees": 6},
])
out = analyze(df, models=models).to_dict()          # JSON-ready
print(out["entities"][0]["jurisdiction"], out["entities"][0]["risk_level"], out["entities"][0]["ml_drivers"])

# 4. Pillar Two table.
print(res.pillar2[["group", "jurisdiction", "safe_harbour", "estimated_topup_tax"]].head())

# 5. Research mode on the public CbCR data.
r = research.run()
print(r.by_year[["year", "reports", "haven_share_profit", "haven_share_employees"]])
research.write_outputs(r, "results/research")
