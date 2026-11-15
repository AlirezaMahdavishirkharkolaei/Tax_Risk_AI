"""Pillar Two (GloBE) screening from CbCR data.

Per group, fiscal year and jurisdiction this module

* blends all entities in the jurisdiction (GloBE ETRs are jurisdictional),
* runs the three tests of the **Transitional CbCR Safe Harbour**
  (OECD Administrative Guidance on Safe Harbours and Penalty Relief, Dec. 2022;
  extended by one year in the January 2026 Side-by-Side Package):
    - de minimis: CbCR revenue < EUR 10m and profit before tax < EUR 1m,
    - simplified ETR: tax expense / profit before tax >= transition rate
      (15% for fiscal years starting in 2023-2024, 16% in 2025, 17% in 2026-2027),
    - routine profits: profit before tax <= substance-based income exclusion (SBIE),
* estimates the top-up tax: (15% - ETR) x (profit - SBIE).

This is a screening estimate, not a GloBE computation: CbCR profit is not GloBE
income, CbCR tax is not adjusted covered taxes, and the safe harbour formally
requires figures from qualified financial statements. Amounts are assumed to
be in EUR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics

MINIMUM_RATE = 0.15
DE_MINIMIS_REVENUE = 10_000_000
DE_MINIMIS_PROFIT = 1_000_000

# SBIE transitional percentages by the year a fiscal year starts
# (Art. 9.2 GloBE Model Rules / Art. 48 Directive (EU) 2022/2523): payroll starts at
# 10% falling 0.2pp a year then 0.8pp; tangible assets start at 8% falling 0.2pp then 0.4pp.
SBIE_RATES = {
    2023: (0.098, 0.078), 2024: (0.096, 0.076), 2025: (0.094, 0.074), 2026: (0.092, 0.072),
    2027: (0.090, 0.070), 2028: (0.082, 0.066), 2029: (0.074, 0.062), 2030: (0.066, 0.058),
    2031: (0.058, 0.054), 2032: (0.050, 0.050),
}
LONG_RUN_RATES = (0.05, 0.05)


def sbie_rates(year: int) -> tuple[float, float]:
    """(payroll %, tangible-asset %) for a fiscal year starting in ``year``."""
    if year < 2023:
        return SBIE_RATES[2023]
    return SBIE_RATES.get(year, LONG_RUN_RATES)


def transition_rate(year: int) -> float | None:
    """Simplified-ETR threshold of the transitional CbCR safe harbour (None = no longer available)."""
    if year <= 2024:
        return 0.15
    if year == 2025:
        return 0.16
    if year in (2026, 2027):
        return 0.17
    return None


def assess(df: pd.DataFrame, fiscal_year: int = 2026) -> pd.DataFrame:
    """Jurisdictional Pillar Two screening. ``df`` is a loaded dataset (see metrics.load_dataset)."""
    d = df.copy()
    d["_tax"] = metrics.tax_for_etr(d)
    if "year" not in d.columns:
        d["year"] = fiscal_year
    for col in ("tangible_assets", "payroll_costs"):
        if col not in d.columns:
            d[col] = np.nan

    agg = d.groupby(["group", "year", "jurisdiction"], sort=False).agg(
        entities=("revenue", "size"),
        revenue=("revenue", "sum"),
        profit_before_tax=("profit_before_tax", "sum"),
        tax=("_tax", "sum"),
        employees=("employees", "sum"),
        tangible_assets=("tangible_assets", lambda s: s.sum(min_count=1)),
        payroll_costs=("payroll_costs", lambda s: s.sum(min_count=1)),
    ).reset_index()

    rows = []
    for _, r in agg.iterrows():
        year = int(r["year"])
        p_rate, a_rate = sbie_rates(year)
        pbt, tax = r["profit_before_tax"], r["tax"]
        have_payroll = not pd.isna(r["payroll_costs"])
        have_assets = not pd.isna(r["tangible_assets"])
        sbie = p_rate * (r["payroll_costs"] if have_payroll else 0) + a_rate * (r["tangible_assets"] if have_assets else 0)
        sbie_complete = have_payroll and have_assets

        etr = tax / pbt if pbt > 0 else np.nan
        rate = transition_rate(year)

        de_minimis = bool(r["revenue"] < DE_MINIMIS_REVENUE and pbt < DE_MINIMIS_PROFIT)
        if rate is None:
            simplified_etr = None
        else:
            simplified_etr = bool(pbt > 0 and etr >= rate)
        if pbt <= 0 or pbt <= sbie:
            routine = True  # a partial SBIE already covers the profit, so the full one does too
        elif sbie_complete:
            routine = False
        else:
            routine = None  # cannot tell without payroll / tangible assets

        tests = [de_minimis, simplified_etr, routine]
        if rate is None:
            safe_harbour = "Not available"
        elif any(t is True for t in tests):
            safe_harbour = "Applies"
        elif all(t is False for t in tests):
            safe_harbour = "Fails"
        else:
            safe_harbour = "Undetermined"

        excess = max(0.0, pbt - sbie)
        topup_pct = max(0.0, MINIMUM_RATE - etr) if pbt > 0 else 0.0
        topup = 0.0 if safe_harbour == "Applies" else topup_pct * excess

        rows.append({
            "group": r["group"], "year": year, "jurisdiction": r["jurisdiction"],
            "entities": int(r["entities"]), "revenue": r["revenue"], "profit_before_tax": pbt,
            "tax": tax, "simplified_etr": etr, "transition_rate": rate,
            "sbie": sbie, "sbie_complete": sbie_complete,
            "test_de_minimis": de_minimis, "test_simplified_etr": simplified_etr, "test_routine_profits": routine,
            "safe_harbour": safe_harbour, "topup_percentage": topup_pct, "excess_profit": excess,
            "estimated_topup_tax": topup,
            # Without payroll the SBIE is understated, so the estimate is an upper bound.
            "estimate_is_upper_bound": (not sbie_complete) and topup > 0,
        })
    return pd.DataFrame(rows)


def group_totals(p2: pd.DataFrame) -> pd.DataFrame:
    return p2.groupby(["group", "year"]).agg(
        jurisdictions=("jurisdiction", "size"),
        safe_harbour_applies=("safe_harbour", lambda s: int((s == "Applies").sum())),
        estimated_topup_tax=("estimated_topup_tax", "sum"),
    ).reset_index()
