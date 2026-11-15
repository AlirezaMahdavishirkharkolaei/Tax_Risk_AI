# Model card: TaxRiskScan default classifier

| | |
|---|---|
| Version | TaxRiskScan 3.0.0 |
| Task | Score how closely a CbCR row's financial profile resembles that of multinationals' tax-haven affiliates |
| Model | logistic regression on cubic-spline features (class-balanced) |
| Anomaly detector | k-means (k=3) on ordinary entities, risky direction only |
| Training data | `training_public_cbcr.csv`: EU Tax Observatory Public CbCR database, 5,004 rows (96 multinationals, 237 reports, 2015–2023) |
| Label | jurisdiction classified as a tax haven in the EU Tax Observatory database (811 positive rows, 16.2%) |
| Inputs | `low_tax_gap`, `log_ppe_ratio`, `log_revenue_ratio`, `profit_margin`, `profit_activity_gap`, `employee_share`, `log_employees`, `loss_making` |
| Not used as inputs | Jurisdiction names, EU lists and zero-tax lists. These would leak the label |
| Validation | 5-fold GroupKFold by multinational |
| ROC AUC | **0.734** |
| Training data SHA-256 | `018a8dc2803d82eca31037ef2f1bbd5ec2c8c8bb946d9e34876039ab47be78ec` |
| Data terms | EU Tax Observatory Public CbCR database: free for non-commercial use with citation (Aliprandi, Borders, Gabriel & von Zedlitz, 2022). Commercial users should train on their own data. |

## Intended use

Use it to prioritise CbCR rows for human review, in research, teaching, audit planning and compliance self-assessment. It is one of three signals in the TaxRiskScan score and is weighted at 30%.

## Out of scope

Do not use it for:

* automated decisions about individual taxpayers
* estimates of tax revenue loss
* legal conclusions about avoidance or abuse

## Performance and limitations

* **Modest separation:** an AUC of about 0.73 separates the classes only moderately. The label is a jurisdiction classification, not an audit outcome, and many haven affiliates (for example in the Netherlands, Belgium or Singapore) are real operating businesses.
* **Selective sample:** public CbCRs are published voluntarily by large, often European groups, so the training sample is selective.
* **Hard cases:** entities with genuinely high margins, such as software or IP owners, or with legitimate low ETRs from incentives or loss relief, can score high.
* **Weaker for some groups:** performance is lower for groups with few jurisdictions, because group shares are then less informative.

## Fairness and responsible use

The model scores companies, not people. Its main risk is over-flagging jurisdictions whose legitimate activity looks like a haven profile.

* Keep a human reviewer in every decision.
* Report the configuration fingerprint and this card with results.
* Retrain on your own labelled outcomes where possible: `taxriskscan train cases.csv --save model.joblib`.

## Explanations

Each scored row lists its main **model drivers**: the change in probability when a feature is set to the typical ordinary value.
