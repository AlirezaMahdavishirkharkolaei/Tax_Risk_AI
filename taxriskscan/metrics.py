"""Load a CbCR-style dataset and compute the financial indicators used for scoring."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from . import jurisdictions

REQUIRED_COLUMNS = ["jurisdiction", "revenue", "profit_before_tax", "tax_paid", "employees"]
OPTIONAL_COLUMNS = ["group", "tangible_assets", "label"]

# Accepted alternative column names (lower-case) -> canonical name.
COLUMN_ALIASES = {
    "country": "jurisdiction",
    "tax_jurisdiction": "jurisdiction",
    "turnover": "revenue",
    "total_revenues": "revenue",
    "profit": "profit_before_tax",
    "profit_loss_before_income_tax": "profit_before_tax",
    "pbt": "profit_before_tax",
    "income_tax_paid": "tax_paid",
    "tax": "tax_paid",
    "number_of_employees": "employees",
    "emps": "employees",
    "tangible_assets_other_than_cash": "tangible_assets",
    "mne_group": "group",
    "company": "group",
}


class DataError(ValueError):
    """Raised when an input dataset cannot be analysed."""


def load_dataset(source: str | Path | IO | pd.DataFrame) -> pd.DataFrame:
    """Read a CSV (path, file object or DataFrame) and return a validated, clean frame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        try:
            df = pd.read_csv(source)
        except Exception as exc:  # pragma: no cover - pandas error messages vary
            raise DataError(f"Could not read CSV: {exc}") from exc

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected: " + ", ".join(REQUIRED_COLUMNS)
        )
    if df.empty:
        raise DataError("The dataset has no rows.")

    numeric = ["revenue", "profit_before_tax", "tax_paid", "employees", "tangible_assets"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[["revenue", "profit_before_tax", "tax_paid", "employees"]].isna().any(axis=1)
    if bad.any():
        rows = ", ".join(str(i + 2) for i in df.index[bad][:10])  # +2: header + 1-based
        raise DataError(f"Non-numeric or empty values in required columns (CSV rows: {rows}).")
    if (df["employees"] < 0).any():
        raise DataError("'employees' cannot be negative.")

    if "group" not in df.columns:
        df["group"] = "Group"
    df["group"] = df["group"].fillna("Group").astype(str)
    df["jurisdiction"] = df["jurisdiction"].astype(str).map(jurisdictions.normalise)
    return df.reset_index(drop=True)


def _share(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Each row's share of its group's (non-negative) total."""
    positive = values.clip(lower=0)
    totals = positive.groupby(groups).transform("sum")
    return (positive / totals.replace(0, np.nan)).fillna(0.0)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add effective tax rate, productivity and profit-shifting indicators."""
    out = df.copy()
    profit = out["profit_before_tax"]
    profitable = profit > 0

    # ETR is only meaningful for profitable entities; losses get NaN, not a fake 0%.
    out["etr"] = np.where(profitable, out["tax_paid"] / profit.where(profitable), np.nan)
    out["profit_margin"] = np.where(out["revenue"] > 0, profit / out["revenue"].where(out["revenue"] > 0), np.nan)
    out["profit_per_employee"] = np.where(out["employees"] > 0, profit / out["employees"].where(out["employees"] > 0), np.nan)
    out["revenue_per_employee"] = np.where(out["employees"] > 0, out["revenue"] / out["employees"].where(out["employees"] > 0), np.nan)

    # Profit vs. substance inside each group (the CbCR "mismatch" test).
    out["profit_share"] = _share(profit, out["group"])
    out["employee_share"] = _share(out["employees"], out["group"])
    if "tangible_assets" in out.columns and out["tangible_assets"].notna().any():
        out["asset_share"] = _share(out["tangible_assets"].fillna(0), out["group"])
        out["activity_share"] = (out["employee_share"] + out["asset_share"]) / 2
    else:
        out["activity_share"] = out["employee_share"]
    out["profit_activity_gap"] = out["profit_share"] - out["activity_share"]

    out["eu_non_cooperative"] = out["jurisdiction"].map(jurisdictions.is_eu_non_cooperative)
    out["zero_cit_jurisdiction"] = out["jurisdiction"].map(jurisdictions.is_zero_or_nominal_cit)
    return out


def feature_matrix(ind: pd.DataFrame, minimum_rate: float = 0.15) -> pd.DataFrame:
    """Numeric features for the ML models (no NaNs, comparable across datasets)."""
    etr = ind["etr"]
    profitable = ind["profit_before_tax"] > 0
    # Shortfall below the GloBE minimum; losses carry no shortfall.
    low_tax_gap = np.where(profitable, np.clip(minimum_rate - etr.fillna(minimum_rate), 0, minimum_rate), 0.0)
    ppe = ind["profit_per_employee"]
    # Profitable entity with no staff: treat as extreme productivity.
    ppe = np.where(profitable & ind["employees"].eq(0), 5_000_000, ppe.fillna(0))
    return pd.DataFrame(
        {
            "low_tax_gap": low_tax_gap / minimum_rate,
            "log_profit_per_employee": np.sign(ppe) * np.log10(1 + np.abs(ppe)),
            "profit_margin": ind["profit_margin"].fillna(0).clip(-1, 1),
            "profit_activity_gap": ind["profit_activity_gap"],
            "eu_non_cooperative": ind["eu_non_cooperative"].astype(float),
            "zero_cit_jurisdiction": ind["zero_cit_jurisdiction"].astype(float),
        },
        index=ind.index,
    )
