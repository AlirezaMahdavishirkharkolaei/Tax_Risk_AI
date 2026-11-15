# Changelog

## 3.0.0 — 2026-11-15

Ready to adopt in organisations, universities and research.

* **Organisations:** TOML configuration for thresholds, weights, risk levels, currency conversion and the organisation's own watchlist
* **Data quality:** checks for aggregate rows, unknown jurisdiction names, tax above profit, negative revenue and duplicates
* **Audit trail:** tool version, input SHA-256, configuration fingerprint and model-data hash in every report and in the JSON output
* **Saved models:** `train --save` writes a model plus a model card; `--model` reuses it
* **Batch analysis** of many files, and `--json` output
* **REST API** (FastAPI) with API keys and OpenAPI docs, plus a Dockerfile and docker-compose for self-hosting
* **Research mode** (`taxriskscan research`): haven shares of profit, employees and tax by year; ETRs; leading havens; sectors; Pillar Two exposure; out-of-sample validation. Writes a research brief
* **Web app:** research section and data-quality messages
* **Documentation:** CITATION.cff, model card, methodology, organisation guide, API guide, examples

## 2.0.0

* Model trained on real public CbCR data (EU Tax Observatory)
* Pillar Two screening
* Legal basis for every rule
* Per-entity explanations
* OECD benchmarks
* OECD CbC XML import
* Browser web app (Pyodide)

## 1.0.0

* Rule engine, logistic regression, k-means, CLI, Streamlit dashboard and HTML report, replacing the original 48-line prototype
