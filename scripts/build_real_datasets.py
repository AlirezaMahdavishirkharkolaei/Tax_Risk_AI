"""Build TaxRiskScan's real-data files from two public sources.

1. EU Tax Observatory - Public Country-by-Country Reports database
   (Aliprandi, Borders, Gabriel & von Zedlitz, 2022), as compiled with Data For
   Good for the Taxplorer tool. Entity-level public CbCRs of 100+ multinationals,
   2014-2023, amounts converted to EUR, each jurisdiction tagged as tax haven or
   not. Terms: free for non-commercial use with citation (see DATA_SOURCES.md).
   Mirror used: github.com/dataforgoodfr/12_taxobservatory
   (commit 966b359a66c963ddeb031612ee897b2e0aa81708),
   file dataviz/data/data_final_dataviz.csv

2. OECD - Corporate Tax Statistics, aggregated CbCR Table I, fiscal year 2016
   (stats.oecd.org, dataset CBCR_TABLEI), amounts in USD.
   Mirror used: github.com/fheitmueller/cbcrdata
   (commit 79a7eb4790604aecc8058db099a35c2a868a0c52), file data_raw/CBCR_TABLEI.csv

Usage:
    git clone https://github.com/dataforgoodfr/12_taxobservatory
    git clone https://github.com/fheitmueller/cbcrdata
    python scripts/build_real_datasets.py \
        --taxobs 12_taxobservatory/dataviz/data/data_final_dataviz.csv \
        --oecd cbcrdata/data_raw/CBCR_TABLEI.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "taxriskscan" / "data"


def build_training(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    # Drop residual "Other / Rest of world" rows: they are not jurisdictions.
    d = d[d["jur_code"].str.upper() != "OTHER"]
    d = d.dropna(subset=["profit_before_tax", "employees", "total_revenues"])
    d = d[d["tax_paid"].notna() | d["tax_accrued"].notna()]
    d = d[(d["total_revenues"] >= 0) & (d["employees"] >= 0)]

    out = pd.DataFrame(
        {
            "group": d["mnc"].str.strip(),
            "year": d["year"].astype(int),
            "jurisdiction": d["jur_name"].str.strip(),
            "jur_code": d["jur_code"],
            "sector": d["sector"],
            "upe_jurisdiction": d["upe_name"],
            "revenue": d["total_revenues"],
            "profit_before_tax": d["profit_before_tax"],
            "tax_paid": d["tax_paid"],
            "tax_accrued": d["tax_accrued"],
            "employees": d["employees"],
            "tangible_assets": d["tangible_assets"],
            "label": d["jur_tax_haven"].astype(bool).astype(int),
        }
    )
    # Keep reports with at least 3 jurisdictions so within-group shares mean something.
    size = out.groupby(["group", "year"])["jurisdiction"].transform("size")
    out = out[size >= 3]
    return out.sort_values(["group", "year", "revenue"], ascending=[True, True, False]).reset_index(drop=True)


def build_benchmarks(path: Path) -> pd.DataFrame:
    c = pd.read_csv(path, encoding="utf-8-sig")
    excluded = {"Foreign Jurisdictions Total", "Stateless", "Other Groups",
                "Other Africa", "Other Americas", "Other Asia", "Other Europe"}
    c = c[~c["Partner Jurisdiction"].isin(excluded)]
    wide = c.pivot_table(index=["COU", "JUR", "Partner Jurisdiction", "PAN"], columns="CBC",
                         values="Value", aggfunc="sum").reset_index()

    def totals(panel: str) -> pd.DataFrame:
        w = wide[wide["PAN"] == panel]
        return w.groupby(["JUR", "Partner Jurisdiction"])[["PROFIT", "TAX_ACCRUED", "EMPLOYEES", "TOT_REV", "CBCR_COUNT"]].sum(min_count=1)

    allg, pos = totals("PANELA"), totals("PANELAI")
    b = pd.DataFrame(index=allg.index)
    b["n_reports"] = allg["CBCR_COUNT"]
    b["etr_profitable"] = pos["TAX_ACCRUED"] / pos["PROFIT"].where(pos["PROFIT"] > 0)
    b["profit_per_employee_usd"] = allg["PROFIT"] / allg["EMPLOYEES"].where(allg["EMPLOYEES"] > 0)
    gp, ge = allg["PROFIT"].clip(lower=0).sum(), allg["EMPLOYEES"].sum()
    b["profit_to_employee_share_ratio"] = (allg["PROFIT"].clip(lower=0) / gp) / (allg["EMPLOYEES"] / ge)
    b = b.reset_index().rename(columns={"JUR": "jur_code", "Partner Jurisdiction": "jurisdiction"})
    b = b[b["n_reports"].fillna(0) >= 5].replace([np.inf, -np.inf], np.nan)
    return b.sort_values("jurisdiction").round(4).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build real training and benchmark data.")
    p.add_argument("--taxobs", type=Path, required=True)
    p.add_argument("--oecd", type=Path, required=True)
    a = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t = build_training(a.taxobs)
    t.to_csv(OUT_DIR / "training_public_cbcr.csv", index=False)
    print(f"training_public_cbcr.csv: {len(t)} rows, {t['group'].nunique()} multinationals, "
          f"{t.groupby(['group', 'year']).ngroups} reports, {int(t['label'].sum())} tax-haven rows, "
          f"years {t['year'].min()}-{t['year'].max()}")

    b = build_benchmarks(a.oecd)
    b.to_csv(OUT_DIR / "oecd_cbcr_2016_benchmarks.csv", index=False)
    print(f"oecd_cbcr_2016_benchmarks.csv: {len(b)} jurisdictions")


if __name__ == "__main__":
    main()
