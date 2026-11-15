"""Tunable thresholds and weights. Every number used for scoring lives here."""

from __future__ import annotations

from dataclasses import dataclass, field


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


DEFAULT_CONFIG = Config()
