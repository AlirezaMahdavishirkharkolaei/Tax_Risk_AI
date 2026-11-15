"""Data-quality checks run before scoring.

Problems in CbCR data are common (aggregate rows, unit errors, typos in
country names). Each check returns a plain-language warning so a reviewer can
fix the input or read the results with the right caution. Checks never change
the data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from . import benchmarks, jurisdictions

AGGREGATE_WORDS = ("stateless", "other", "rest of", "total", "unallocated", "various", "eliminations")


@dataclass(frozen=True)
class Issue:
    level: str  # "warning" or "info"
    check: str
    message: str
    rows: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rows"] = list(self.rows)
        return d


def _rows(mask: pd.Series) -> tuple[int, ...]:
    # +2: 1-based and header line, so numbers match the CSV a user opens.
    return tuple(int(i) + 2 for i in mask[mask].index[:20])


def _known_names() -> set[str]:
    from .cbcxml import ISO2
    names = {n.casefold() for n in ISO2.values()}
    names |= {n.casefold() for n in jurisdictions.EU_ANNEX_I | jurisdictions.ZERO_OR_NOMINAL_CIT}
    names |= set(benchmarks.table().index)
    return names


def check(df: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    j = df["jurisdiction"].astype(str)
    pbt, tax = df["profit_before_tax"], df["tax_paid"]
    if "tax_accrued" in df.columns:
        tax = df["tax_accrued"].fillna(tax)

    agg = j.str.casefold().map(lambda s: any(w in s for w in AGGREGATE_WORDS))
    if agg.any():
        issues.append(Issue("warning", "aggregate_rows",
                            f"{int(agg.sum())} row(s) look like aggregates rather than jurisdictions "
                            f"({', '.join(sorted(set(j[agg]))[:5])}). They distort group shares; consider removing them.",
                            _rows(agg)))

    known = _known_names()
    unknown = ~agg & ~j.map(lambda s: benchmarks._key(s) in known)
    if unknown.any():
        issues.append(Issue("info", "unknown_jurisdictions",
                            "Jurisdiction name(s) not recognised - check spelling so list and benchmark checks work: "
                            + ", ".join(sorted(set(j[unknown]))[:8]), _rows(unknown)))

    over = (pbt > 0) & (tax > pbt * 1.0)
    if over.any():
        issues.append(Issue("warning", "tax_exceeds_profit",
                            f"{int(over.sum())} row(s) report tax above profit (ETR > 100%). Check units or signs.",
                            _rows(over)))

    neg_rev = df["revenue"] < 0
    if neg_rev.any():
        issues.append(Issue("warning", "negative_revenue", f"{int(neg_rev.sum())} row(s) have negative revenue.",
                            _rows(neg_rev)))

    margin = (df["revenue"] > 0) & (pbt > df["revenue"] * 1.5)
    if margin.any():
        issues.append(Issue("warning", "profit_exceeds_revenue",
                            f"{int(margin.sum())} row(s) report profit well above revenue. Dividends from "
                            "subsidiaries can cause this; check whether they should be excluded.", _rows(margin)))

    keys = ["group", "year", "jurisdiction"] if "year" in df.columns else ["group", "jurisdiction"]
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        issues.append(Issue("info", "duplicate_jurisdictions",
                            f"{int(dup.sum())} rows share group, year and jurisdiction. They are scored as separate "
                            "entities and blended for Pillar Two.", _rows(dup)))

    for col, why in (("payroll_costs", "the Pillar Two routine-profits test may be undetermined"),
                     ("tangible_assets", "substance is measured from employees only")):
        if col not in df.columns or df[col].isna().all():
            issues.append(Issue("info", f"missing_{col}", f"No '{col}' column: {why}."))

    if "group" in df.columns and (df["group"] == "Group").all() and len(df) > 1:
        issues.append(Issue("info", "single_group", "No 'group' column: all rows are treated as one group."))
    return issues
