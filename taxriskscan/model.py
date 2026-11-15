"""Machine-learning signals: a logistic-regression classifier and k-means anomaly detection.

Training data
-------------
By default both models are trained on **real public CbCR data**: the EU Tax
Observatory's Public Country-by-Country Reports database (~5,000 jurisdiction
rows from ~100 multinationals, 2015-2023). The label is whether the
jurisdiction is classified as a tax haven in that database. The classifier
therefore estimates how much an entity's *financial profile* (low tax, profit
per employee, profit far from staff, ...) resembles that of multinationals'
tax-haven affiliates - without ever seeing the jurisdiction's name.

A synthetic dataset (``scripts/generate_synthetic_data.py``) is kept for tests
and demos, and you can train on your own labelled cases.

Models
------
* **Classifier** - logistic regression on a cubic-spline expansion of the
  features (captures thresholds such as "ETR near zero" while staying a linear
  model), with class weighting for the imbalanced labels.
* **Anomaly detector** - k-means is fitted on ordinary (label 0) entities using
  only features where higher always means riskier. Values below the ordinary
  median are floored to it, so an entity is never anomalous for paying *more*
  tax or earning *less* per employee. The score is the share of ordinary
  entities lying closer to their cluster centre, rescaled so a perfectly
  typical entity scores 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from . import metrics
from .config import Config, DEFAULT_CONFIG

REAL_TRAINING = "training_public_cbcr.csv"
SYNTHETIC_TRAINING = "training_cbcr.csv"

FEATURE_LABELS = {
    "low_tax_gap": "low effective tax rate",
    "log_ppe_ratio": "high profit per employee vs. group",
    "log_revenue_ratio": "high revenue per employee vs. group",
    "profit_margin": "profit margin",
    "profit_activity_gap": "profit share above activity share",
    "employee_share": "share of group staff",
    "log_employees": "number of employees",
    "loss_making": "loss-making",
}


def data_path(name: str) -> Path:
    return Path(str(resources.files("taxriskscan") / "data" / name))


def default_training_path() -> Path:
    return data_path(REAL_TRAINING)


def resolve_training(training) -> object:
    """Accept None / 'real' / 'synthetic' / a path / a DataFrame."""
    if training is None or (isinstance(training, str) and training == "real"):
        return default_training_path()
    if isinstance(training, str) and training == "synthetic":
        return data_path(SYNTHETIC_TRAINING)
    return training


def _classifier():
    return make_pipeline(
        StandardScaler(),
        SplineTransformer(n_knots=5, degree=3),
        LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"),
    )


@dataclass
class RiskModels:
    classifier: object
    scaler: StandardScaler
    kmeans: KMeans
    reference_distances: np.ndarray
    floor: np.ndarray
    baseline: pd.Series  # median ordinary entity, used for explanations
    feature_names: list[str]
    anomaly_features: list[str]
    training_source: str
    n_training: int
    n_positive: int
    cv_auc: float | None = None

    # -- classifier --------------------------------------------------------
    def probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X[self.feature_names])[:, 1]

    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-entity contribution of each feature to the probability.

        Contribution = probability(actual) - probability(with that feature set to the
        typical ordinary value). Positive values push the entity towards "haven-like".
        """
        base = self.probability(X)
        out = {}
        for f in self.feature_names:
            Xf = X[self.feature_names].copy()
            Xf[f] = self.baseline[f]
            out[f] = base - self.probability(Xf)
        return pd.DataFrame(out, index=X.index)

    def importance(self, X: pd.DataFrame) -> pd.Series:
        return self.explain(X).abs().mean().sort_values(ascending=False)

    # -- anomaly -------------------------------------------------------------
    def _z(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.scaler.transform(X[self.anomaly_features]), self.floor)

    def _ecdf(self, d: np.ndarray) -> np.ndarray:
        ref = self.reference_distances
        return np.searchsorted(ref, d, side="right") / len(ref)

    def anomaly(self, X: pd.DataFrame) -> np.ndarray:
        d = self.kmeans.transform(self._z(X)).min(axis=1)
        base = self._ecdf(self.kmeans.transform(self.floor[None, :]).min(axis=1))[0]
        return np.clip((self._ecdf(d) - base) / max(1e-9, 1 - base), 0, 1)

    def cluster(self, X: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(self._z(X))


def _prepare(training, config: Config):
    src = resolve_training(training)
    df = metrics.load_dataset(src)
    if "label" not in df.columns:
        raise metrics.DataError("Training data needs a 'label' column (1 = high risk, 0 = ordinary).")
    y = pd.to_numeric(df["label"], errors="coerce")
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise metrics.DataError("'label' must contain only 0 and 1.")
    if y.nunique() < 2:
        raise metrics.DataError("Training data needs both classes (0 and 1).")
    X = metrics.feature_matrix(metrics.compute_indicators(df), config.thresholds.globe_minimum_rate)
    name = "DataFrame" if isinstance(src, pd.DataFrame) else Path(str(src)).name
    return df, X, y.astype(int), name


def train(training=None, config: Config = DEFAULT_CONFIG) -> RiskModels:
    """Fit both models. ``training``: None/'real', 'synthetic', a CSV path or a DataFrame with ``label``."""
    df, X, y, name = _prepare(training, config)
    names = list(metrics.FEATURES)

    cv_auc = None
    if min(np.bincount(y)) >= 5 and df["group"].nunique() >= 5:
        # Split by multinational so one group never sits in both train and test folds.
        cv = GroupKFold(n_splits=5)
        cv_auc = float(cross_val_score(_classifier(), X[names], y, groups=df["group"], cv=cv, scoring="roc_auc").mean())
    clf = _classifier().fit(X[names], y)

    anomaly_features = list(metrics.RISK_DIRECTION_FEATURES)
    normal = X.loc[y == 0, anomaly_features]
    scaler = StandardScaler().fit(normal)
    Zn = scaler.transform(normal)
    floor = np.median(Zn, axis=0)
    Zn = np.maximum(Zn, floor)
    k = max(1, min(config.n_clusters, len(normal)))
    km = KMeans(n_clusters=k, n_init=10, random_state=config.random_state).fit(Zn)
    ref = np.sort(km.transform(Zn).min(axis=1))

    baseline = X.loc[y == 0, names].median()
    return RiskModels(clf, scaler, km, ref, floor, baseline, names, anomaly_features,
                      name, len(df), int(y.sum()), cv_auc)


def evaluate_on(models: RiskModels, df: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> float:
    """ROC AUC of the classifier on a labelled dataset."""
    ind = metrics.compute_indicators(metrics.load_dataset(df))
    X = metrics.feature_matrix(ind, config.thresholds.globe_minimum_rate)
    return float(roc_auc_score(ind["label"].astype(int), models.probability(X)))
