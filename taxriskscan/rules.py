"""Transparent, rule-based risk indicators.

Every rule produces points, a human-readable reason and a documentation
recommendation, so each flag in the final report can be traced back to a test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Thresholds


@dataclass(frozen=True)
class Flag:
    code: str
    points: int
    reason: str
    recommendation: str


MAX_POINTS = 12  # cap used to normalise the rule score to 0-1


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def evaluate_row(row: pd.Series, t: Thresholds, watchlist: frozenset[str] = frozenset()) -> list[Flag]:
    flags: list[Flag] = []
    profit = row["profit_before_tax"]
    etr = row["etr"]

    if profit > 0 and not np.isnan(etr):
        if etr < t.very_low_etr:
            flags.append(Flag(
                "ETR_VERY_LOW", 3,
                f"Effective tax rate {etr:.1%} is below {t.very_low_etr:.0%}.",
                "Obtain the local tax computation and reconcile tax paid to the statutory rate; "
                "check for incentives, rulings or IP regimes.",
            ))
        elif etr < t.globe_minimum_rate:
            flags.append(Flag(
                "ETR_BELOW_GLOBE", 2,
                f"Effective tax rate {etr:.1%} is below the {t.globe_minimum_rate:.0%} Pillar Two minimum.",
                "Run a GloBE ETR calculation for the jurisdiction and assess exposure to a "
                "top-up tax (QDMTT / IIR / UTPR).",
            ))

    ppe = row["profit_per_employee"]
    if profit > 0 and row["employees"] == 0:
        flags.append(Flag(
            "NO_EMPLOYEES", 3,
            "Reports profit with no employees (possible shell / conduit entity).",
            "Document the entity's substance: board meetings, decision makers, premises and "
            "outsourced functions.",
        ))
    elif not np.isnan(ppe):
        if ppe > t.very_high_profit_per_employee:
            flags.append(Flag(
                "PPE_VERY_HIGH", 2,
                f"Profit per employee {_fmt_money(ppe)} exceeds {_fmt_money(t.very_high_profit_per_employee)}.",
                "Review transfer-pricing documentation (functional analysis) for the entity.",
            ))
        elif ppe > t.high_profit_per_employee:
            flags.append(Flag(
                "PPE_HIGH", 1,
                f"Profit per employee {_fmt_money(ppe)} exceeds {_fmt_money(t.high_profit_per_employee)}.",
                "Check that the profit level is consistent with the functions performed.",
            ))

    gap = row["profit_activity_gap"]
    if gap > t.strong_profit_activity_mismatch:
        flags.append(Flag(
            "PROFIT_SUBSTANCE_STRONG", 3,
            f"Holds {row['profit_share']:.0%} of group profit but only {row['activity_share']:.0%} "
            "of its economic activity.",
            "Prioritise for transfer-pricing audit: profit allocation is far from the location of "
            "staff and assets.",
        ))
    elif gap > t.profit_activity_mismatch:
        flags.append(Flag(
            "PROFIT_SUBSTANCE", 2,
            f"Profit share ({row['profit_share']:.0%}) exceeds activity share "
            f"({row['activity_share']:.0%}) by {gap:.0%}.",
            "Review intra-group pricing (royalties, interest, service fees) flowing to this entity.",
        ))

    margin = row["profit_margin"]
    if not np.isnan(margin) and margin > t.very_high_margin:
        flags.append(Flag(
            "MARGIN_VERY_HIGH", 1,
            f"Pre-tax margin of {margin:.0%} is unusually high.",
            "Compare with benchmarking studies for comparable independent companies.",
        ))

    if row["eu_non_cooperative"]:
        flags.append(Flag(
            "EU_ANNEX_I", 3,
            "Jurisdiction is on the EU list of non-cooperative jurisdictions (Annex I).",
            "Check EU defensive measures (e.g. CFC, withholding, non-deductibility) and DAC6 "
            "reporting obligations.",
        ))
    if watchlist and str(row["jurisdiction"]).casefold() in watchlist:
        flags.append(Flag(
            "WATCHLIST", 2,
            "Jurisdiction is on your organisation's watchlist.",
            "Apply the review procedure your organisation defines for watchlisted jurisdictions.",
        ))
    if row["zero_cit_jurisdiction"]:
        flags.append(Flag(
            "ZERO_CIT", 1,
            "Jurisdiction levies no or nominal general corporate income tax.",
            "Confirm whether a domestic minimum top-up tax applies to the group.",
        ))
    return flags


def evaluate(ind: pd.DataFrame, t: Thresholds, watchlist=()) -> pd.DataFrame:
    """Return rule points (0-MAX_POINTS), normalised score and flag details per row."""
    from .jurisdictions import normalise

    wl = frozenset(normalise(j).casefold() for j in watchlist)
    all_flags = [evaluate_row(row, t, wl) for _, row in ind.iterrows()]
    points = [min(sum(f.points for f in flags), MAX_POINTS) for flags in all_flags]
    return pd.DataFrame(
        {
            "rule_points": points,
            "rule_score": [p / MAX_POINTS for p in points],
            "flags": [[f.code for f in flags] for flags in all_flags],
            "reasons": [[f.reason for f in flags] for flags in all_flags],
            "recommendations": [[f.recommendation for f in flags] for flags in all_flags],
        },
        index=ind.index,
    )
