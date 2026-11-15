# TaxRiskScan: AI System for Detecting High-Risk Tax Jurisdictions

[![tests](https://github.com/AlirezaMahdavishirkharkolaei/Tax_Risk_AI/actions/workflows/tests.yml/badge.svg)](https://github.com/AlirezaMahdavishirkharkolaei/Tax_Risk_AI/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/code-MIT-green)

TaxRiskScan analyses multinational company data in Country-by-Country Reporting (CbCR) format. It flags jurisdictions that show signs of tax avoidance, low taxation or aggressive profit shifting, and it screens each jurisdiction for **Pillar Two** global minimum tax exposure.

It combines three things:

* **transparent legal-economic rules**, each linked to its **legal basis** (EU directives, OECD guidelines)
* a **logistic-regression classifier trained on real public CbCRs** from ~100 multinationals
* **k-means anomaly detection**

Every flag comes with a plain-language reason, a recommended review step and the provisions to consult.

*Started as a project for the Building AI course (Reaktor & University of Helsinki), now a working tool.*

![Risk heatmap](docs/heatmap.png)

> **Risk indicators, not legal conclusions.** A high score means "look closer", not "non-compliant". Low effective tax rates and high profits can have legitimate explanations, such as loss relief, incentives, timing differences or genuine returns on IP. Every flag needs human review.

## What it does

| | |
|---|---|
| **Risk scoring** | A 0–100 score per entity: 50% rules + 30% ML probability + 20% anomaly |
| **Real-data model** | Trained on the EU Tax Observatory's Public CbCR database (5,004 rows, 96 groups, 2015–2023) |
| **Explainability** | For each entity, which features pushed the model's probability up |
| **Pillar Two** | Transitional CbCR Safe Harbour tests (de minimis, simplified ETR, routine profits) and an estimated top-up tax per jurisdiction |
| **Legal basis** | Each flag linked to ATAD, the Pillar Two Directive, DAC6, DAC4, the public CbCR Directive, the OECD TPG and BEPS Actions |
| **Benchmarks** | Each jurisdiction's aggregate ETR from OECD CbCR statistics, as context |
| **Inputs** | CSV, or **OECD CbC XML** (schema v2.0) |
| **Outputs** | Heatmap, score chart, summary CSV, Pillar Two CSV, and a self-contained HTML audit report (print to PDF) |
| **Interfaces** | Command line and a Streamlit web dashboard |

## Quick start

```bash
git clone https://github.com/AlirezaMahdavishirkharkolaei/Tax_Risk_AI.git
cd Tax_Risk_AI
pip install -e ".[app,dev]"        # or: pip install -r requirements.txt

taxriskscan analyze data/sample_cbcr.csv --out results/   # CSV
taxriskscan analyze data/sample_cbc.xml --out results/    # OECD CbC XML
taxriskscan train                                         # model diagnostics
streamlit run app.py                                      # web dashboard
pytest                                                    # 30 tests
```

`python Tax_Risk_AI.py` still works. It analyses the bundled sample and writes the report to `results/`.

### Deploy the dashboard (free)

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), choose **New app**, select this repository, and set the main file to `app.py`.
3. `requirements.txt` and `.streamlit/config.toml` are already set up.

## Input data

One row per entity or jurisdiction (CbCR Table 1):

| column | required | meaning |
|---|---|---|
| `jurisdiction` | ✓ | country / territory |
| `revenue` | ✓ | total revenues |
| `profit_before_tax` | ✓ | profit (loss) before income tax |
| `tax_paid` / `tax_accrued` | one of them | income tax paid (cash) / current tax accrued. ETR uses the accrued figure when given |
| `employees` | ✓ | number of employees |
| `group`, `year` | | multinational group and fiscal year. Shares are computed per group and year |
| `tangible_assets` | | tangible assets other than cash. Used for substance and the SBIE |
| `payroll_costs` | | needed for a complete Pillar Two routine-profits test |

Common alternative headers (`country`, `turnover`, `profit`, …) are recognised. Run `taxriskscan sample` to print a template.

## Real training data

The classifier is trained on the **EU Tax Observatory's Public Country-by-Country Reports database**: CbCRs that ~100 large multinationals published voluntarily, standardised and converted to EUR. The label is whether the jurisdiction is classified as a **tax haven** in that database.

Jurisdiction names and lists are deliberately **not** model inputs. The model only sees currency-free financial ratios: low-tax gap, profit and revenue per employee relative to the group, margin, profit share vs. share of staff and assets, and size. So it learns what a *haven-like financial profile* looks like, and can flag one in any jurisdiction.

| | |
|---|---|
| Training rows | 5,004 jurisdictions · 96 multinationals · 237 reports · 2015–2023 |
| Positive (tax-haven) rows | 811 |
| Validation | 5-fold **group-wise** cross-validation (a multinational never appears in both train and test) |
| ROC AUC | **0.73** |

The AUC is modest by design. The label is noisy, since many affiliates in Belgium, the Netherlands or Singapore are real operating businesses. That is exactly why the ML signal is blended with transparent rules rather than used alone. On the bundled synthetic data the same model reaches 0.99, which shows how optimistic synthetic benchmarks are.

Use your own labelled cases (e.g. past audit outcomes):

```bash
taxriskscan train my_cases.csv
taxriskscan analyze data.csv --train my_cases.csv
```

Sources, citations and terms of use are in **[DATA_SOURCES.md](DATA_SOURCES.md)**. The real training data is free for **non-commercial** use with citation.

## Pillar Two screening

For each group, year and jurisdiction (entities blended, as the GloBE rules require):

* **De minimis test:** CbCR revenue < €10m and profit before tax < €1m
* **Simplified ETR test:** tax / profit ≥ transition rate (15% for 2023–24, 16% for 2025, 17% for 2026–27)
* **Routine profits test:** profit ≤ substance-based income exclusion (SBIE), using the transitional payroll and tangible-asset percentages
* **Estimated top-up tax** = (15% − ETR) × (profit − SBIE) where the safe harbour does not apply

This is a screening estimate from CbCR figures, not a full GloBE computation. If payroll costs are missing, the routine-profits test may be *undetermined* and the top-up estimate becomes an upper bound.

![Risk scores](docs/risk_scores.png)

## Methods

| method | role | where |
|---|---|---|
| Rule-based scoring | explainable flags with reasons, recommendations and legal basis | `rules.py`, `legal.py` |
| Logistic regression (spline features) | probability of a haven-like profile, trained on real CbCRs | `model.py` |
| k-means clustering | anomaly score relative to ordinary entities, counting deviations in the risky direction only | `model.py` |
| Feature contributions | per-entity explanation of the ML probability | `model.py` |
| Pillar Two engine | safe-harbour tests, SBIE, top-up estimate | `pillar2.py` |

All thresholds and weights are in [`taxriskscan/config.py`](taxriskscan/config.py).

## Project structure

```
taxriskscan/
  metrics.py        load & validate data, indicators, ML features
  rules.py          rule engine (reasons + recommendations)
  legal.py          legal basis for each rule
  model.py          logistic regression, k-means, explanations
  pillar2.py        Transitional CbCR Safe Harbour + top-up estimate
  benchmarks.py     OECD 2016 jurisdiction benchmarks
  cbcxml.py         OECD CbC XML import
  pipeline.py       puts it all together
  report.py         heatmap, charts, HTML report
  jurisdictions.py  EU Annex I and zero-tax lists
  data/             real training data, OECD benchmarks, synthetic data
app.py              Streamlit dashboard
scripts/            rebuild real data, generate synthetic data
tests/              pytest suite
```

## Challenges and limitations

* Produces risk indicators only. It does not give legal conclusions.
* Public CbCRs come from multinationals that chose to publish, so they are not a random sample. The tax-haven label is a proxy for risk, not a finding of abuse.
* CbCR "tax" is not GloBE adjusted covered taxes, and CbCR profit is not GloBE income.
* Pillar Two thresholds assume amounts in EUR.
* The OECD benchmarks are from 2016, and the jurisdiction lists change twice a year.
* Anomalies may be legitimate and require human review.

## What next?

* Newer OECD aggregated CbCR years and EU public CbCR filings (Directive 2021/2101) as training data
* QDMTT / IIR / UTPR allocation of the top-up tax
* AI-written executive summary of the audit report
* Automatic updates of the EU list

## Acknowledgments

* Building AI course by Reaktor & University of Helsinki
* EU Tax Observatory (Public CbCR database) and Data For Good
* OECD Corporate Tax Statistics
* Academic literature on tax transparency and profit shifting

## License

Code: [MIT](LICENSE). Bundled data: see [DATA_SOURCES.md](DATA_SOURCES.md).
