"""Load a CbCR-style dataset and compute the financial indicators used for scoring."""

from __future__ import annotations

from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from . import jurisdictions

REQUIRED_COLUMNS = ["jurisdiction", "revenue", "profit_before_tax", "tax_paid", "employees"]
OPTIONAL_COLUMNS = ["group", "year", "tax_accrued", "tangible_assets", "payroll_costs", "label"]
# Either tax column satisfies the "tax" requirement.
TAX_COLUMNS = ("tax_paid", "tax_accrued")

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
    "income_tax_paid_on_cash_basis": "tax_paid",
    "tax": "tax_paid",
    "income_tax_accrued": "tax_accrued",
    "income_tax_accrued_current_year": "tax_accrued",
    "fiscal_year": "year",
    "payroll": "payroll_costs",
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

    if "tax_paid" not in df.columns and "tax_accrued" in df.columns:
        df["tax_paid"] = np.nan
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(
            "Missing required column(s): " + ", ".join(missing)
            + ". Expected: " + ", ".join(REQUIRED_COLUMNS)
        )
    if df.empty:
        raise DataError("The dataset has no rows.")

    numeric = ["revenue", "profit_before_tax", "tax_paid", "tax_accrued", "employees",
               "tangible_assets", "payroll_costs", "year"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    tax_missing = df["tax_paid"].isna()
    if "tax_accrued" in df.columns:
        tax_missing &= df["tax_accrued"].isna()
    bad = df[["revenue", "profit_before_tax", "employees"]].isna().any(axis=1) | tax_missing
    if bad.any():
        rows = ", ".join(str(i + 2) for i in df.index[bad][:10])  # +2: header + 1-based
        raise DataError(f"Non-numeric or empty values in required columns (CSV rows: {rows}).")
    if (df["employees"] < 0).any():
        raise DataError("'employees' cannot be negative.")

    if "group" not in df.columns:
        df["group"] = "Group"
    df["group"] = df["group"].fillna("Group").astype(str)
    if "year" in df.columns:
        if df["year"].isna().any():
            raise DataError("'year' must be filled in for every row (or the column removed).")
        df["year"] = df["year"].astype(int)
    df["jurisdiction"] = df["jurisdiction"].astype(str).map(jurisdictions.normalise)
    return df.reset_index(drop=True)


def group_keys(df: pd.DataFrame) -> list[str]:
    """A 'report' is one group in one year (or one group if there is no year)."""
    return ["group", "year"] if "year" in df.columns else ["group"]


def tax_for_etr(df: pd.DataFrame) -> pd.Series:
    """Current tax accrued when available (closer to GloBE), otherwise cash tax paid."""
    if "tax_accrued" in df.columns:
        return df["tax_accrued"].fillna(df["tax_paid"])
    return df["tax_paid"]


def _share(values: pd.Series, df: pd.DataFrame) -> pd.Series:
    """Each row's share of its report's (non-negative) total."""
    positive = values.clip(lower=0)
    totals = positive.groupby([df[k] for k in group_keys(df)]).transform("sum")
    return (positive / totals.replace(0, np.nan)).fillna(0.0)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add effective tax rate, productivity and profit-shifting indicators."""
    out = df.copy()
    profit = out["profit_before_tax"]
    profitable = profit > 0

    # ETR is only meaningful for profitable entities; losses get NaN, not a fake 0%.
    out["etr_tax"] = tax_for_etr(out)
    out["etr"] = np.where(profitable, out["etr_tax"] / profit.where(profitable), np.nan)
    out["profit_margin"] = np.where(out["revenue"] > 0, profit / out["revenue"].where(out["revenue"] > 0), np.nan)
    out["profit_per_employee"] = np.where(out["employees"] > 0, profit / out["employees"].where(out["employees"] > 0), np.nan)
    out["revenue_per_employee"] = np.where(out["employees"] > 0, out["revenue"] / out["employees"].where(out["employees"] > 0), np.nan)

    # Profit vs. substance inside each group (the CbCR "mismatch" test).
    out["profit_share"] = _share(profit, out)
    out["employee_share"] = _share(out["employees"], out)
    activity = out["employee_share"]
    if "tangible_assets" in out.columns and out["tangible_assets"].notna().any():
        # Use assets only for reports that disclose them.
        out["asset_share"] = _share(out["tangible_assets"].fillna(0), out)
        has_assets = out.groupby(group_keys(out))["tangible_assets"].transform(lambda s: s.notna().any())
        activity = np.where(has_assets, (out["employee_share"] + out["asset_share"]) / 2, activity)
    out["activity_share"] = activity
    out["profit_activity_gap"] = out["profit_share"] - out["activity_share"]

    # Profit per employee relative to the whole report (currency-free).
    keys = [out[k] for k in group_keys(out)]
    g_profit = profit.clip(lower=0).groupby(keys).transform("sum")
    g_emp = out["employees"].groupby(keys).transform("sum")
    group_ppe = g_profit / g_emp.replace(0, np.nan)
    out["ppe_ratio"] = np.where(
        profitable & group_ppe.gt(0),
        profit.where(profitable) / out["employees"].clip(lower=1) / group_ppe,
        0.0,
    )
    g_rev = out["revenue"].clip(lower=0).groupby(keys).transform("sum")
    group_rpe = g_rev / g_emp.replace(0, np.nan)
    out["revenue_ratio"] = (out["revenue"].clip(lower=0) / out["employees"].clip(lower=1) / group_rpe).fillna(1.0)

    out["eu_non_cooperative"] = out["jurisdiction"].map(jurisdictions.is_eu_non_cooperative)
    out["zero_cit_jurisdiction"] = out["jurisdiction"].map(jurisdictions.is_zero_or_nominal_cit)
    return out


FEATURES = [
    "low_tax_gap", "log_ppe_ratio", "log_revenue_ratio", "profit_margin",
    "profit_activity_gap", "employee_share", "log_employees", "loss_making",
]
# Features where a higher value always means "riskier" (used for anomaly detection).
RISK_DIRECTION_FEATURES = ["low_tax_gap", "log_ppe_ratio", "log_revenue_ratio", "profit_margin", "profit_activity_gap"]


def feature_matrix(ind: pd.DataFrame, minimum_rate: float = 0.15) -> pd.DataFrame:
    """Currency-free numeric features for the ML models.

    Jurisdiction lists are deliberately *not* features: the real training labels
    are tax-haven jurisdictions, so list membership would leak the label. The
    models learn what a haven-like *financial profile* looks like.
    """
    etr = ind["etr"]
    profitable = ind["profit_before_tax"] > 0
    low_tax_gap = np.where(profitable, np.clip(minimum_rate - etr.fillna(minimum_rate), 0, minimum_rate), 0.0)
    return pd.DataFrame(
        {
            "low_tax_gap": low_tax_gap / minimum_rate,
            "log_ppe_ratio": np.log10(1 + ind["ppe_ratio"].clip(lower=0, upper=1e5)),
            "log_revenue_ratio": np.log10(ind["revenue_ratio"].clip(lower=1e-3, upper=1e5)),
            "profit_margin": ind["profit_margin"].fillna(0).clip(-1, 1),
            "profit_activity_gap": ind["profit_activity_gap"],
            "employee_share": ind["employee_share"],
            "log_employees": np.log10(1 + ind["employees"]),
            "loss_making": (~profitable).astype(float),
        },
        index=ind.index,
    )
