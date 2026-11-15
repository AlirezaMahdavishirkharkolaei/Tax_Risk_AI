"""Jurisdiction benchmarks from OECD aggregated CbCR statistics (fiscal year 2016).

For every partner jurisdiction with at least 5 reports, across all reporting
countries: the effective tax rate of profitable sub-groups (tax accrued /
profit) and the ratio of the jurisdiction's share of profit to its share of
employees (above 1 = more profit than staff would suggest).
Source: OECD Corporate Tax Statistics, CBCR_TABLEI (see DATA_SOURCES.md).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from . import jurisdictions
from .model import data_path

FILE = "oecd_cbcr_2016_benchmarks.csv"

# OECD spellings -> names used elsewhere in TaxRiskScan
_OECD_NAMES = {
    "china (people's republic of)": "china",
    "korea": "south korea",
    "russia": "russian federation",
    "virgin islands (british)": "british virgin islands",
    "hong kong, china": "hong kong",
    "united states of america": "united states",
    "türkiye": "turkey",
    "chinese taipei": "taiwan",
}


def _key(name: str) -> str:
    k = jurisdictions.normalise(name).casefold()
    return _OECD_NAMES.get(k, k)


@lru_cache(maxsize=1)
def table() -> pd.DataFrame:
    b = pd.read_csv(data_path(FILE))
    b["key"] = b["jurisdiction"].map(_key)
    return b.drop_duplicates("key").set_index("key")


def lookup(names: pd.Series) -> pd.DataFrame:
    t = table()
    keys = names.map(_key)
    return pd.DataFrame(
        {
            "oecd_etr_2016": keys.map(t["etr_profitable"]),
            "oecd_profit_employee_ratio_2016": keys.map(t["profit_to_employee_share_ratio"]),
        },
        index=names.index,
    )
