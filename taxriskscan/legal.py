"""Legal and regulatory basis for each risk flag.

Each rule in ``rules.py`` is linked to the instruments a reviewer would turn to.
References are to the instrument and provision; EU acts link to EUR-Lex.
This is orientation for further research, not legal advice.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    instrument: str
    provision: str
    url: str | None = None

    def label(self) -> str:
        return f"{self.instrument}, {self.provision}" if self.provision else self.instrument


EURLEX = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:"

PILLAR2_DIR = "Council Directive (EU) 2022/2523 (Pillar Two Directive)"
ATAD = "Council Directive (EU) 2016/1164 (ATAD)"
DAC6 = "Council Directive (EU) 2018/822 (DAC6)"
DAC4 = "Council Directive (EU) 2016/881 (DAC4)"
PUBLIC_CBCR = "Directive (EU) 2021/2101 (public CbCR)"
TPG = "OECD Transfer Pricing Guidelines (2022)"
GLOBE = "OECD GloBE Model Rules (2021)"

URLS = {
    PILLAR2_DIR: EURLEX + "32022L2523",
    ATAD: EURLEX + "32016L1164",
    DAC6: EURLEX + "32018L0822",
    DAC4: EURLEX + "32016L0881",
    PUBLIC_CBCR: EURLEX + "32021L2101",
    "EU list": "https://www.consilium.europa.eu/en/policies/eu-list-of-non-cooperative-jurisdictions/",
}


def _s(instrument: str, provision: str = "") -> Source:
    return Source(instrument, provision, URLS.get(instrument))


LEGAL_BASIS: dict[str, list[Source]] = {
    "ETR_VERY_LOW": [
        _s(ATAD, "Art. 7 (controlled foreign company rules: low actual tax test)"),
        _s(PILLAR2_DIR, "Arts. 26-27 (effective tax rate and top-up tax)"),
    ],
    "ETR_BELOW_GLOBE": [
        _s(PILLAR2_DIR, "Arts. 26-28 (ETR, top-up tax, substance-based income exclusion)"),
        _s(GLOBE, "Arts. 5.1-5.3"),
        _s("OECD Administrative Guidance on Safe Harbours (Dec. 2022)", "Transitional CbCR Safe Harbour"),
    ],
    "NO_EMPLOYEES": [
        _s(ATAD, "Art. 7(2)(a) (substantive economic activity carve-out)"),
        _s("OECD BEPS Action 5", "substantial activity requirement"),
    ],
    "PPE_VERY_HIGH": [
        _s(TPG, "Ch. I (arm's length principle) and Ch. VI (intangibles, DEMPE functions)"),
        _s("OECD BEPS Actions 8-10", "aligning transfer pricing outcomes with value creation"),
    ],
    "PPE_HIGH": [_s(TPG, "Ch. I (accurate delineation of the transaction)")],
    "PROFIT_SUBSTANCE_STRONG": [
        _s(TPG, "Ch. I, VI and IX (risk, intangibles, business restructurings)"),
        _s("OECD BEPS Action 13", "CbCR as a high-level risk-assessment tool"),
        _s(DAC4, "automatic exchange of CbC reports"),
    ],
    "PROFIT_SUBSTANCE": [
        _s(TPG, "Ch. VII (intra-group services) and Ch. X (financial transactions)"),
        _s("OECD BEPS Action 13", "CbCR as a high-level risk-assessment tool"),
    ],
    "MARGIN_VERY_HIGH": [_s(TPG, "Ch. II-III (methods, comparability analysis)")],
    "EU_ANNEX_I": [
        _s("EU list", "Council conclusions of 9 October 2026 (Annex I)"),
        _s(DAC6, "Annex IV, hallmark C.1(b)(ii)"),
        _s(PUBLIC_CBCR, "new Art. 48c of Directive 2013/34/EU (separate disclosure for listed jurisdictions)"),
    ],
    "WATCHLIST": [Source("Organisation watchlist", "configured under [lists] watchlist")],
    "ZERO_CIT": [
        _s(DAC6, "Annex IV, hallmark C.1(b)(i) (zero or almost-zero corporate tax)"),
        _s(PILLAR2_DIR, "Art. 11 (qualified domestic top-up tax)"),
    ],
}


def basis_for(code: str) -> list[Source]:
    return LEGAL_BASIS.get(code, [])
