"""Generate the synthetic, labelled reference dataset used to train the models.

Real CbCR data is published only in aggregated form (OECD CbCR statistics,
EU Tax Observatory), without entity-level "profit shifting" labels. To train
and test the classifier we therefore simulate multinational groups with a
documented, reproducible story:

* **Ordinary entities (label 0)** - staff and revenue drawn from realistic
  ranges, modest margins, effective tax rates scattered around the local
  statutory rate. About 15% are loss-making. A minority enjoy legitimate low
  ETRs (incentives, loss relief) so low tax alone is not a perfect signal.
* **Profit-shifting hubs (label 1)** - few staff, a large share of group profit,
  low ETRs, frequently (not always) located in low-tax jurisdictions. About a
  quarter are given real staff, ordinary margins and moderate ETRs, and some
  ordinary entities are genuinely high-margin, so the classes overlap.

Labels come from the simulated scenario, *not* from TaxRiskScan's rules, so the
classifier is not just re-learning the rule engine.

Usage:  python scripts/generate_synthetic_data.py [--groups 120] [--seed 7]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Approximate headline statutory CIT rates - only used to simulate data.
ORDINARY = {
    "Germany": 0.30, "France": 0.25, "Italy": 0.24, "Spain": 0.25, "United Kingdom": 0.25,
    "United States": 0.25, "Canada": 0.26, "Japan": 0.30, "Australia": 0.30, "Brazil": 0.34,
    "India": 0.25, "Mexico": 0.30, "Poland": 0.19, "Sweden": 0.21, "Turkey": 0.25,
    "South Africa": 0.27, "China": 0.25, "South Korea": 0.24, "Austria": 0.23, "Belgium": 0.25,
}
HUBS = {
    "Ireland": 0.125, "Netherlands": 0.258, "Luxembourg": 0.24, "Singapore": 0.17,
    "Switzerland": 0.14, "Cyprus": 0.125, "Malta": 0.35, "Hong Kong": 0.165,
    "Bermuda": 0.0, "Cayman Islands": 0.0, "British Virgin Islands": 0.0, "Jersey": 0.0,
    "Bahamas": 0.0, "Isle of Man": 0.0, "Guernsey": 0.0, "Anguilla": 0.0,
    "Turks and Caicos Islands": 0.0, "US Virgin Islands": 0.21, "Vanuatu": 0.0, "Palau": 0.0,
}


def simulate(n_groups: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        group = f"SimGroup-{g:03d}"
        n_ord = rng.integers(4, 10)
        n_hub = rng.choice([0, 1, 1, 2])
        ordinary = rng.choice(list(ORDINARY), size=n_ord, replace=False)
        group_profit = 0.0

        for j in ordinary:
            emp = int(rng.lognormal(5.0, 1.0)) + 5
            rev = emp * rng.uniform(120_000, 450_000)
            loss = rng.random() < 0.15
            if loss:
                margin = -rng.uniform(0.01, 0.15)
            elif rng.random() < 0.10:  # genuinely high-margin business (software, IP owner)
                margin = rng.uniform(0.2, 0.55)
            else:
                margin = max(0.005, rng.normal(0.08, 0.05))
            profit = rev * margin
            if loss:
                tax = rev * rng.uniform(0, 0.002)  # minimum taxes / withholding
            elif rng.random() < 0.12:  # legitimate incentive / loss relief
                tax = profit * rng.uniform(0.03, 0.13)
            else:
                tax = profit * max(0.0, rng.normal(ORDINARY[j], 0.05))
            assets = rev * rng.uniform(0.1, 0.6)
            rows.append((group, j, rev, profit, tax, emp, assets, 0))
            group_profit += max(profit, 0)

        for j in rng.choice(list(HUBS), size=n_hub, replace=False):
            substantive = rng.random() < 0.25
            emp = int(rng.integers(40, 400)) if substantive else int(rng.choice([0, 1, 2, 3, 5, 8, 12, 25]))
            profit = group_profit * (rng.uniform(0.1, 0.6) if substantive else rng.uniform(0.3, 2.5))
            rev = profit / (rng.uniform(0.08, 0.3) if substantive else rng.uniform(0.2, 0.9))
            etr = rng.uniform(0.08, 0.15) if substantive else rng.uniform(0.0, min(0.08, HUBS[j] + 0.02))
            tax = profit * etr
            assets = rev * rng.uniform(0.0, 0.05)
            rows.append((group, j, rev, profit, tax, emp, assets, 1))

    df = pd.DataFrame(rows, columns=[
        "group", "jurisdiction", "revenue", "profit_before_tax", "tax_paid",
        "employees", "tangible_assets", "label",
    ])
    money = ["revenue", "profit_before_tax", "tax_paid", "tangible_assets"]
    df[money] = df[money].round(0)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--groups", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parents[1] / "taxriskscan" / "data" / "training_cbcr.csv")
    a = p.parse_args()
    df = simulate(a.groups, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} entities ({int(df.label.sum())} hubs) from {a.groups} groups to {a.out}")


if __name__ == "__main__":
    main()
