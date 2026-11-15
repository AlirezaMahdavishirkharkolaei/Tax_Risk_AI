"""Tunable thresholds and weights. Every number used for scoring lives here.

Organisations can override any value with a TOML file (see
``taxriskscan.example.toml`` and :func:`load_config`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True)
class Thresholds:
    # Effective tax rate (tax paid / profit before tax), profitable entities only.
    globe_minimum_rate: float = 0.15  # OECD Pillar Two (GloBE) minimum rate
    very_low_etr: float = 0.10
    # Profit per employee (in the dataset's currency).
    high_profit_per_employee: float = 100_000
    very_high_profit_per_employee: float = 300_000
    # Profit share minus economic-activity share inside a group (percentage points).
    profit_activity_mismatch: float = 0.10
    strong_profit_activity_mismatch: float = 0.25
    # Pre-tax profit margin (profit / revenue).
    very_high_margin: float = 0.50


@dataclass(frozen=True)
class Weights:
    """How the three signals are blended into the final 0-100 risk score."""

    rules: float = 0.5
    classifier: float = 0.3
    clustering: float = 0.2


@dataclass(frozen=True)
class Levels:
    medium: float = 35.0
    high: float = 60.0


@dataclass(frozen=True)
class Config:
    thresholds: Thresholds = field(default_factory=Thresholds)
    weights: Weights = field(default_factory=Weights)
    levels: Levels = field(default_factory=Levels)
    n_clusters: int = 3
    random_state: int = 42
    # Organisation settings
    organisation: str = ""
    # Amounts in the input are multiplied by eur_rate to express them in EUR
    # (Pillar Two thresholds and profit-per-employee limits are in EUR).
    currency: str = "EUR"
    eur_rate: float = 1.0
    # Jurisdictions the organisation wants flagged in addition to the EU list.
    watchlist: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["watchlist"] = list(self.watchlist)
        return d

    def fingerprint(self) -> str:
        """Short hash of every setting, printed in reports for reproducibility."""
        raw = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()[:12]


DEFAULT_CONFIG = Config()


class ConfigError(ValueError):
    pass


def _section(cls, values: dict, name: str):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown setting(s) in [{name}]: {', '.join(sorted(unknown))}")
    try:
        return cls(**{k: float(v) for k, v in values.items()})
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid value in [{name}]: {exc}") from exc


def config_from_dict(d: dict) -> Config:
    allowed = {"thresholds", "weights", "levels", "organisation", "data", "lists", "model"}
    unknown = set(d) - allowed
    if unknown:
        raise ConfigError(f"Unknown section(s): {', '.join(sorted(unknown))}")
    org = d.get("organisation", {})
    data = d.get("data", {})
    lists = d.get("lists", {})
    mdl = d.get("model", {})
    weights = _section(Weights, d.get("weights", {}), "weights")
    if min(weights.rules, weights.classifier, weights.clustering) < 0 or \
            weights.rules + weights.classifier + weights.clustering <= 0:
        raise ConfigError("Weights must be non-negative and not all zero.")
    levels = _section(Levels, d.get("levels", {}), "levels")
    if not 0 <= levels.medium <= levels.high <= 100:
        raise ConfigError("Levels must satisfy 0 <= medium <= high <= 100.")
    rate = float(data.get("eur_rate", 1.0))
    if rate <= 0:
        raise ConfigError("[data] eur_rate must be positive.")
    return Config(
        thresholds=_section(Thresholds, d.get("thresholds", {}), "thresholds"),
        weights=weights,
        levels=levels,
        n_clusters=int(mdl.get("n_clusters", 3)),
        random_state=int(mdl.get("random_state", 42)),
        organisation=str(org.get("name", "")),
        currency=str(data.get("currency", "EUR")).upper(),
        eur_rate=rate,
        watchlist=tuple(str(x) for x in lists.get("watchlist", [])),
    )


def load_config(path: str | Path) -> Config:
    """Read a TOML configuration file."""
    try:
        with open(path, "rb") as fh:
            return config_from_dict(tomllib.load(fh))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
