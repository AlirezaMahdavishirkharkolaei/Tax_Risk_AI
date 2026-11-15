"""Machine-learning signals: a logistic-regression classifier and k-means anomaly detection.

Both models are trained on a reference dataset of labelled entities
(``taxriskscan/data/training_cbcr.csv`` by default, synthetic - see
``scripts/generate_synthetic_data.py``). Training takes well under a second, so
models are fitted on demand rather than shipped as pickles.

* **Classifier** - logistic regression on the indicator features; outputs the
  probability that an entity looks like a profit-shifting hub.
* **Anomaly detector** - k-means is fitted on the entities labelled as ordinary
  (label 0). An entity's anomaly score is the share of ordinary training
  entities that sit closer to their cluster centre than it does (0 = typical,
  1 = further out than every ordinary entity). All features point in the
  "riskier" direction, and values below the ordinary median are floored to it,
  so an entity is never anomalous merely for paying more tax or earning less
  per employee than usual.
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
from sklearn.preprocessing import StandardScaler

from . import metrics
from .config import Config, DEFAULT_CONFIG


def default_training_path() -> Path:
    return Path(str(resources.files("taxriskscan") / "data" / "training_cbcr.csv"))


@dataclass
class RiskModels:
    classifier: object
    scaler: StandardScaler
    kmeans: KMeans
    reference_distances: np.ndarray
    floor: np.ndarray
    feature_names: list[str]
    cv_auc: float | None = None

    # -- scoring -----------------------------------------------------------
    def probability(self, X: pd.DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(X[self.feature_names])[:, 1]

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        return np.maximum(self.scaler.transform(X[self.feature_names]), self.floor)

    def anomaly(self, X: pd.DataFrame) -> np.ndarray:
        d = self.kmeans.transform(self._z(X)).min(axis=1)
        return self._rescale(self._ecdf(d))

    def _ecdf(self, d: np.ndarray) -> np.ndarray:
        ref = self.reference_distances
        return np.searchsorted(ref, d, side="right") / len(ref)

    def _rescale(self, p: np.ndarray) -> np.ndarray:
        # A perfectly typical entity (every feature at the ordinary median) maps to 0.
        base = self._ecdf(self.kmeans.transform(self.floor[None, :]).min(axis=1))[0]
        return np.clip((p - base) / max(1e-9, 1 - base), 0, 1)

    def cluster(self, X: pd.DataFrame) -> np.ndarray:
        return self.kmeans.predict(self._z(X))

    def coefficients(self) -> pd.Series:
        lr = self.classifier[-1]
        return pd.Series(lr.coef_[0], index=self.feature_names).sort_values(key=abs, ascending=False)


def train(training: str | Path | pd.DataFrame | None = None, config: Config = DEFAULT_CONFIG) -> RiskModels:
    """Fit both models. ``training`` must contain the usual columns plus ``label`` (0/1)."""
    df = metrics.load_dataset(training if training is not None else default_training_path())
    if "label" not in df.columns:
        raise metrics.DataError("Training data needs a 'label' column (1 = high risk, 0 = ordinary).")
    y = pd.to_numeric(df["label"], errors="coerce")
    if y.isna().any() or not set(y.unique()) <= {0, 1}:
        raise metrics.DataError("'label' must contain only 0 and 1.")
    if y.nunique() < 2:
        raise metrics.DataError("Training data needs both classes (0 and 1).")
    y = y.astype(int)

    X = metrics.feature_matrix(metrics.compute_indicators(df), config.thresholds.globe_minimum_rate)
    names = list(X.columns)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    cv_auc = None
    if min(np.bincount(y)) >= 5 and df["group"].nunique() >= 5:
        # Split by group so entities of one multinational never sit in both folds.
        cv = GroupKFold(n_splits=5)
        cv_auc = float(cross_val_score(clf, X, y, groups=df["group"], cv=cv, scoring="roc_auc").mean())
    clf.fit(X, y)

    normal = X[y == 0]
    scaler = StandardScaler().fit(normal)
    Zn = scaler.transform(normal)
    floor = np.median(Zn, axis=0)
    Zn = np.maximum(Zn, floor)
    k = max(1, min(config.n_clusters, len(normal)))
    km = KMeans(n_clusters=k, n_init=10, random_state=config.random_state).fit(Zn)
    ref = np.sort(km.transform(Zn).min(axis=1))

    return RiskModels(clf, scaler, km, ref, floor, names, cv_auc)


def evaluate_on(models: RiskModels, df: pd.DataFrame, config: Config = DEFAULT_CONFIG) -> float:
    """ROC AUC of the classifier on a labelled dataset (for reporting)."""
    X = metrics.feature_matrix(metrics.compute_indicators(df), config.thresholds.globe_minimum_rate)
    return float(roc_auc_score(df["label"].astype(int), models.probability(X)))
