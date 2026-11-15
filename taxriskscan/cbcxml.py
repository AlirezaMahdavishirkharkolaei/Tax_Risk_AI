"""Import OECD CbC XML (schema v2.0, ``urn:oecd:ties:cbc:v2``) into a TaxRiskScan dataset.

Reads Table 1 (``CbcReports/Summary``) for each jurisdiction. Namespaces and
prefixes are ignored, so files from different filing systems parse alike.
Only the standard library is used, and DTDs / entities are rejected.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import IO

import pandas as pd

from .metrics import DataError


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child(el, name):
    if el is None:
        return None
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _path(el, *names):
    for n in names:
        el = _child(el, n)
    return el


def _num(el) -> float | None:
    if el is None or el.text is None or not el.text.strip():
        return None
    return float(el.text.strip().replace(",", ""))


def read_cbc_xml(source: str | Path | IO[bytes]) -> pd.DataFrame:
    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    if isinstance(raw, str):
        raw = raw.encode()
    if b"<!DOCTYPE" in raw[:4096] or b"<!ENTITY" in raw:
        raise DataError("XML with DOCTYPE/ENTITY declarations is not accepted.")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise DataError(f"Invalid XML: {exc}") from exc

    year = None
    msg = _child(root, "MessageSpec")
    period = _child(msg, "ReportingPeriod")
    if period is not None and period.text:
        year = int(period.text.strip()[:4])

    rows = []
    for body in (c for c in root if _local(c.tag) == "CbcBody"):
        name_el = _path(body, "ReportingEntity", "Entity", "Name")
        group = name_el.text.strip() if name_el is not None and name_el.text else "Group"
        for rep in (c for c in body if _local(c.tag) == "CbcReports"):
            country = _child(rep, "ResCountryCode")
            s = _child(rep, "Summary")
            if country is None or s is None:
                continue
            ccy = _path(s, "Revenues", "Total")
            rows.append({
                "group": group,
                "jurisdiction": country.text.strip(),
                "revenue": _num(_path(s, "Revenues", "Total")),
                "profit_before_tax": _num(_child(s, "ProfitOrLoss")),
                "tax_paid": _num(_child(s, "TaxPaid")),
                "tax_accrued": _num(_child(s, "TaxAccrued")),
                "employees": _num(_child(s, "NbEmployees")),
                "tangible_assets": _num(_child(s, "Assets")),
                "currency": ccy.get("currCode") if ccy is not None else None,
            })
    if not rows:
        raise DataError("No CbcReports/Summary elements found - is this an OECD CbC XML file?")
    df = pd.DataFrame(rows)
    if year is not None:
        df.insert(1, "year", year)
    df["jurisdiction"] = df["jurisdiction"].map(lambda c: ISO2.get(c, c))
    return df


# ISO 3166-1 alpha-2 codes used in CbC XML -> country names (common CbCR jurisdictions).
ISO2 = {
    "AD": "Andorra", "AE": "United Arab Emirates", "AI": "Anguilla", "AR": "Argentina", "AS": "American Samoa",
    "AT": "Austria", "AU": "Australia", "AW": "Aruba", "BB": "Barbados", "BE": "Belgium", "BG": "Bulgaria",
    "BH": "Bahrain", "BM": "Bermuda", "BR": "Brazil", "BS": "Bahamas", "CA": "Canada", "CH": "Switzerland",
    "CL": "Chile", "CN": "China", "CO": "Colombia", "CW": "Curacao", "CY": "Cyprus", "CZ": "Czech Republic",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "EG": "Egypt", "ES": "Spain", "FI": "Finland",
    "FR": "France", "GB": "United Kingdom", "GG": "Guernsey", "GI": "Gibraltar", "GR": "Greece", "GU": "Guam",
    "HK": "Hong Kong", "HR": "Croatia", "HU": "Hungary", "ID": "Indonesia", "IE": "Ireland", "IL": "Israel",
    "IM": "Isle of Man", "IN": "India", "IS": "Iceland", "IT": "Italy", "JE": "Jersey", "JP": "Japan",
    "KE": "Kenya", "KR": "South Korea", "KY": "Cayman Islands", "LI": "Liechtenstein", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "MA": "Morocco", "MC": "Monaco", "MO": "Macao", "MT": "Malta",
    "MU": "Mauritius", "MX": "Mexico", "MY": "Malaysia", "NG": "Nigeria", "NL": "Netherlands", "NO": "Norway",
    "NZ": "New Zealand", "PA": "Panama", "PE": "Peru", "PH": "Philippines", "PL": "Poland", "PR": "Puerto Rico",
    "PT": "Portugal", "PW": "Palau", "QA": "Qatar", "RO": "Romania", "RS": "Serbia", "RU": "Russian Federation",
    "SA": "Saudi Arabia", "SC": "Seychelles", "SE": "Sweden", "SG": "Singapore", "SI": "Slovenia",
    "SK": "Slovakia", "TC": "Turks and Caicos Islands", "TH": "Thailand", "TR": "Turkey", "TW": "Taiwan",
    "UA": "Ukraine", "US": "United States", "VG": "British Virgin Islands", "VI": "US Virgin Islands",
    "VN": "Viet Nam", "VU": "Vanuatu", "ZA": "South Africa",
}
