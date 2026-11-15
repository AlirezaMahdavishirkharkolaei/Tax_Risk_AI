# Methodology

This note describes how TaxRiskScan turns a country-by-country report into risk indicators. It is written so that a reviewer, a supervisor or a referee can check every step. All thresholds are in [`taxriskscan/config.py`](../taxriskscan/config.py) and can be changed with a configuration file.

## 1. Unit of analysis

One row is one entity or jurisdiction in one multinational group in one fiscal year: CbCR Table 1. Comparisons inside a group are made per **report**, meaning one group in one year.

## 2. Indicators

| Indicator | Definition | Why it matters |
|---|---|---|
| Effective tax rate (ETR) | tax ÷ profit before tax, for profitable rows only. Uses current tax accrued when given, otherwise cash tax paid | Low taxation; Pillar Two |
| Profit per employee | profit ÷ employees | Profit far above what staff could generate |
| Profit–activity gap | share of group profit − share of group activity (mean of employee and tangible-asset shares) | Profit booked away from real activity (BEPS Action 13 risk assessment) |
| Profit per employee vs. group | entity profit per employee ÷ group profit per employee | Currency-free version of the above |
| Revenue per employee vs. group | as above, for revenue | Revenue routed through small entities |
| Margin | profit ÷ revenue | Unusually high returns |

Loss-making entities have **no ETR** and are never penalised for paying no tax. A loss with no tax is economically normal.

## 3. Rule engine

Each rule awards points and produces a reason, a recommended review step and its legal basis ([`legal.py`](../taxriskscan/legal.py)).

| Rule | Points |
|---|---|
| ETR < 10% | 3 |
| 10% ≤ ETR < 15% | 2 |
| Profit with no employees | 3 |
| Profit per employee > €300k | 2 |
| Profit per employee > €100k | 1 |
| Profit–activity gap > 25 pp | 3 |
| Profit–activity gap > 10 pp | 2 |
| Margin > 50% | 1 |
| EU Annex I jurisdiction | 3 |
| Organisation watchlist | 2 |
| Zero or nominal corporate income tax | 1 |

Points are capped at 12 and scaled to 0–1.

## 4. Classifier

* **Training data:** the EU Tax Observatory Public CbCR database. After cleaning there are 5,004 rows, 96 groups and 237 reports, covering 2015–2023.
* **Label:** whether the jurisdiction is classified as a tax haven in that database.
* **Features:** the currency-free indicators above, plus the share of group staff, the log of employees and a loss flag. Jurisdiction names and lists are deliberately excluded so the label cannot leak into the inputs.
* **Model:** logistic regression on a cubic B-spline expansion of the standardised features (5 knots), with L2 penalty C = 0.5 and class-balanced weights.
* **Validation:** 5-fold GroupKFold by multinational, so no group appears in both training and test folds. ROC AUC is about 0.73.
* **Explanations:** for each feature, the change in probability when that feature is set to the median value of ordinary (label 0) rows.

## 5. Anomaly score

k-means (k = 3) is fitted on ordinary rows, using only features where a higher value always means more risk. Values below the ordinary median are floored to it, so an entity is never anomalous merely for paying more tax.

The score is the empirical percentile of the distance to the nearest centroid among ordinary rows, rescaled so that a perfectly typical row scores 0.

## 6. Blended score and levels

score = 100 × (0.5 · rules + 0.3 · probability + 0.2 · anomaly) / (sum of weights)

Levels: **High** ≥ 60, **Medium** ≥ 35, otherwise **Low**.

## 7. Pillar Two screening

Per group, year and jurisdiction, with all entities in the jurisdiction blended:

* **De minimis test:** revenue < €10m and profit < €1m.
* **Simplified ETR test:** tax ÷ profit ≥ 15% for fiscal years starting in 2023–24, 16% in 2025, and 17% in 2026–27.
* **Routine profits test:** profit ≤ SBIE.
* **SBIE:** payroll × payroll rate + tangible assets × asset rate, using the transitional rates by year (2026: 9.2% and 7.2%; both fall to 5% from 2033).
* **Top-up estimate:** where no test is passed, (15% − ETR) × max(0, profit − SBIE).

If payroll is missing, the routine-profits test can be **undetermined** and the estimate is an **upper bound**.

This is a screening approximation. CbCR profit is not GloBE income and CbCR tax is not adjusted covered taxes.

## 8. Research mode

[`research.py`](../taxriskscan/research.py) computes, per year:

* the haven share of positive profit, employees, revenue and tax
* profit-weighted and median ETRs for havens and other jurisdictions
* the leading havens by profit booked
* the haven profit share by sector
* Pillar Two screening outcomes

It also re-evaluates the blended score out of sample. In that check, the classifier's probabilities come from group-wise cross-validation, and the jurisdiction-list rules and the anomaly term are switched off, so the label is never used to predict itself.

## 9. Known limitations

* Public CbCRs are voluntary and therefore selective, and the sample changes from year to year.
* The tax-haven label is a proxy, not a finding of abuse.
* The benchmarks are 2016 OECD aggregates.
* EU lists change twice a year (current list as of 9 October 2026).
* Results are indicators for human review, not legal or tax conclusions.

## 10. Reproducibility

Every report and JSON result records:

* the tool version
* the SHA-256 of the input
* a fingerprint of the configuration
* the SHA-256 of the training data
* library versions

`scripts/build_real_datasets.py` rebuilds the training data from the pinned source commits listed in [DATA_SOURCES.md](../DATA_SOURCES.md).
