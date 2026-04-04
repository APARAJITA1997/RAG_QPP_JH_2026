"""
src/models.py
-------------
QPP Regression Models for RAG-QPP.

Three ensemble tree-based regressors are evaluated:
  1. Random Forest  — best in-domain (Pearson r=0.659 on MS MARCO Passage)
  2. XGBoost        — middle ground across all datasets
  3. LightGBM       — best cross-domain generalisation (ρ=0.476 on NQ)

All models accept a 12-dimensional standardized feature vector x(q)
and predict a scalar retrieval effectiveness score ŷ(q) ≈ MRR@10.

Training uses 5-fold cross-validation with fixed random_state=42
for reproducibility (Section 4).

Reference: Sinha & Chakma (2026), Sections 3.6, 4
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS      = 5


# ══════════════════════════════════════════════════════════════════════════════
# Default hyperparameters (sensible defaults matching paper settings)
# ══════════════════════════════════════════════════════════════════════════════

RF_DEFAULTS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

XGBOOST_DEFAULTS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    tree_method="hist",
)

LGBM_DEFAULTS = dict(
    n_estimators=300,
    max_depth=-1,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


# ══════════════════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════════════════

def _build_rf(**kwargs) -> RandomForestRegressor:
    params = {**RF_DEFAULTS, **kwargs}
    return RandomForestRegressor(**params)


def _build_xgboost(**kwargs):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("xgboost is required:  pip install xgboost")
    params = {**XGBOOST_DEFAULTS, **kwargs}
    return XGBRegressor(**params)


def _build_lightgbm(**kwargs):
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError("lightgbm is required:  pip install lightgbm")
    params = {**LGBM_DEFAULTS, **kwargs}
    return LGBMRegressor(**params)


MODEL_BUILDERS = {
    "random_forest": _build_rf,
    "xgboost":       _build_xgboost,
    "lightgbm":      _build_lightgbm,
}


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: bool = True,
    **model_kwargs,
):
    """
    Train a QPP regression model.

    Parameters
    ----------
    model_name  : "random_forest" | "xgboost" | "lightgbm"
    X_train     : (N, 12) standardized feature matrix.
    y_train     : (N,) ground-truth MRR@10 scores.
    cv          : if True, report 5-fold CV MSE before final fit.
    model_kwargs: override default hyperparameters.

    Returns
    -------
    Fitted sklearn-compatible regressor.
    """
    if model_name not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_BUILDERS)}"
        )

    model = MODEL_BUILDERS[model_name](**model_kwargs)

    if cv:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
        )
        rmse_scores = np.sqrt(-cv_scores)
        logger.info(
            "[%s] %d-fold CV RMSE: %.4f ± %.4f",
            model_name, N_FOLDS, rmse_scores.mean(), rmse_scores.std()
        )

    model.fit(X_train, y_train)
    logger.info("[%s] Training complete on %d samples.", model_name, len(y_train))
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: bool = True,
) -> Dict:
    """Train all three regressors and return {name: fitted_model}."""
    models = {}
    for name in MODEL_BUILDERS:
        logger.info("Training %s …", name)
        models[name] = train_model(name, X_train, y_train, cv=cv)
    return models


# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Return predicted QPP scores ŷ(q) for each query in X.

    Returns
    -------
    np.ndarray of shape (N,).
    """
    return model.predict(X).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved → %s", path)


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded ← %s", path)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Feature importance (Random Forest)
# ══════════════════════════════════════════════════════════════════════════════

def get_feature_importance(
    model,
    feature_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Return {feature_name: importance} from a tree-based model.
    Uses impurity-based importance (Eq. 72 in the paper).
    """
    importances = model.feature_importances_

    if feature_names is None:
        from src.features import FEATURE_NAMES
        feature_names = FEATURE_NAMES

    return dict(zip(feature_names, importances.tolist()))
