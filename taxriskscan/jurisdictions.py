"""Reference lists of jurisdictions used as risk indicators.

These lists change over time. They are kept here as plain data so they can be
reviewed and updated without touching the scoring logic.

Sources
-------
* EU list of non-cooperative jurisdictions for tax purposes (Annex I), as
  revised by the ECOFIN Council on 9 October 2026 (Panama and Viet Nam moved
  to Annex II in that update).
  https://www.consilium.europa.eu/en/policies/eu-list-of-non-cooperative-jurisdictions/
* "Zero / no general corporate income tax" jurisdictions: commonly cited in
  the tax-haven literature as levying no (or a 0% standard) corporate income
  tax. Several of them have introduced a 15% domestic minimum top-up tax for
  large groups under Pillar Two, so membership is only a weak indicator.
"""

from __future__ import annotations

import unicodedata

EU_ANNEX_I_AS_OF = "2026-10-09"

EU_ANNEX_I: frozenset[str] = frozenset(
    {
        "American Samoa",
        "Anguilla",
        "Guam",
        "Palau",
        "Russian Federation",
        "Turks and Caicos Islands",
        "US Virgin Islands",
        "Vanuatu",
    }
)

ZERO_OR_NOMINAL_CIT: frozenset[str] = frozenset(
    {
        "Anguilla",
        "Bahamas",
        "Bahrain",
        "Bermuda",
        "British Virgin Islands",
        "Cayman Islands",
        "Guernsey",
        "Isle of Man",
        "Jersey",
        "Turks and Caicos Islands",
        "Vanuatu",
    }
)

# Common alternative spellings -> canonical name used in the lists above.
_ALIASES = {
    "russia": "Russian Federation",
    "usvi": "US Virgin Islands",
    "u.s. virgin islands": "US Virgin Islands",
    "united states virgin islands": "US Virgin Islands",
    "virgin islands (u.s.)": "US Virgin Islands",
    "bvi": "British Virgin Islands",
    "virgin islands (british)": "British Virgin Islands",
    "cayman": "Cayman Islands",
    "the bahamas": "Bahamas",
    "turks & caicos": "Turks and Caicos Islands",
    "turks and caicos": "Turks and Caicos Islands",
}

_CANONICAL = {name.casefold(): name for name in EU_ANNEX_I | ZERO_OR_NOMINAL_CIT}


def normalise(name: str) -> str:
    """Return the canonical spelling of a jurisdiction name (or the cleaned input)."""
    cleaned = unicodedata.normalize("NFKC", str(name)).strip()
    key = " ".join(cleaned.casefold().split())
    if key in _ALIASES:
        return _ALIASES[key]
    return _CANONICAL.get(key, cleaned)


def is_eu_non_cooperative(name: str) -> bool:
    return normalise(name) in EU_ANNEX_I


def is_zero_or_nominal_cit(name: str) -> bool:
    return normalise(name) in ZERO_OR_NOMINAL_CIT
