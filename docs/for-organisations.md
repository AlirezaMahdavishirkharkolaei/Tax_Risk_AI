# Using TaxRiskScan in an organisation

This guide is for tax administrations, audit and advisory firms, corporate tax teams, universities and research groups.

## Choose a way to run it

| Option | Best for | Data leaves your machine? |
|---|---|---|
| **Web app** (`web/`, GitHub Pages or any static host) | Quick screening, teaching, demos | No: Python runs inside the browser |
| **Command line** (`pip install taxriskscan`) | Analysts, batch jobs, research | No |
| **REST API + Docker** (`docker compose up`) | Integration with case-management or ERP systems, shared team service | No: self-hosted |
| **Streamlit dashboard** (`streamlit run app.py`) | Internal dashboard for a team | Depends where you host it |

## Licensing: pick the right model

The **code** is MIT-licensed, so it can be used commercially. The **bundled real-data model** is trained on the EU Tax Observatory Public CbCR database, which is free for **non-commercial use with citation**.

| Use | Model to use |
|---|---|
| University teaching, academic research, NGOs, public-interest work | Bundled real-data model (default). Cite the database |
| Commercial service or paid advisory work | Train on **your own labelled cases** (`taxriskscan train cases.csv --save model.joblib`), use the synthetic model (`TAXRISKSCAN_TRAINING=synthetic`), or obtain permission from the EU Tax Observatory |

The rule engine, the Pillar Two module, the legal basis, the OECD benchmarks and the jurisdiction lists carry no such restriction.

## Configure it for your policy

```bash
taxriskscan config > org.toml           # example with every setting
taxriskscan analyze data.csv --config org.toml
```

`org.toml` sets:

* your organisation's name, shown on reports
* the input currency and its EUR conversion rate
* every threshold
* the weights of the three signals
* the High and Medium cut-offs
* a **watchlist** of extra jurisdictions, such as a national list of non-cooperative jurisdictions

The report prints a fingerprint of the configuration, so you can tell which settings were used.

## Train on your own cases

If you have past audit outcomes, label them 1 (adjustment or confirmed risk) or 0 (no issue) in a `label` column:

```bash
taxriskscan train cases.csv --save model.joblib     # prints a cross-validated AUC and feature effects
taxriskscan analyze new_data.csv --model model.joblib
```

The model card (`model.joblib.card.json`) records the training data hash, row counts, the AUC and library versions.

**Security:** only load model files your organisation created. joblib files can execute code when loaded.

## REST API

```bash
docker compose up -d                       # or: pip install "taxriskscan[api]" && taxriskscan serve
open http://localhost:8000/docs            # interactive OpenAPI documentation
```

| Endpoint | Purpose |
|---|---|
| `POST /analyze` | Upload a CSV or CbC XML file; returns JSON |
| `POST /analyze/json` | Send entities as JSON |
| `POST /report` | Upload a file; returns the self-contained HTML audit report |
| `POST /pillar2` | Pillar Two screening only |
| `GET /model`, `GET /config` | The model card and the active settings |
| `GET /health` | Liveness check, with no key needed |

Settings are environment variables:

* `TAXRISKSCAN_API_KEYS`: required key(s), sent in the `X-API-Key` header
* `TAXRISKSCAN_CONFIG`
* `TAXRISKSCAN_MODEL`
* `TAXRISKSCAN_TRAINING`
* `TAXRISKSCAN_MAX_UPLOAD_MB`

Uploads are processed in memory and not stored. Put the service behind your own HTTPS reverse proxy.

Example:

```bash
curl -H "X-API-Key: $KEY" -F file=@report.xml http://localhost:8000/analyze
```

## Batch processing

```bash
taxriskscan analyze filings/*.xml --out results/ --config org.toml
```

Each file gets its own folder containing the HTML report, the CSVs, the charts and `result.json`.

## Data quality

Before scoring, TaxRiskScan warns about:

* aggregate rows ("Stateless", "Other", "Total")
* unrecognised jurisdiction names
* tax above profit
* negative revenue
* profit far above revenue (often intra-group dividends)
* duplicate rows
* missing payroll or tangible assets

Warnings appear in the report, the API response and the web app.

## Responsible use

TaxRiskScan ranks cases for review. It does not decide them. A high score is a reason to request documentation, not evidence of avoidance.

* Keep a human reviewer in every decision.
* Record the configuration fingerprint and the model card with each case.
* Revisit thresholds when the law changes, for example the EU list in February and October, or the Pillar Two transition rates.
