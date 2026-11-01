"""Layer 3 - the models.

Every estimator is wrapped in a scikit-learn Pipeline whose first step is the
preprocessor. That is the structural guarantee against data leakage: the
scaler and the one-hot encoder are *fitted* on whatever data is passed to
`.fit()` and merely *applied* by `.predict()`. There is no code path in which
test data can influence the preprocessing.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config
from features import FEATURE_CAT, FEATURE_NUM


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Numeric passthrough or standardisation + one-hot categoricals.

    Scaling is applied only for distance- and gradient-based models (KNN,
    linear regression). Tree ensembles are invariant to feature scale, so
    scaling them would add cost for no benefit.
    """
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, FEATURE_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_CAT),
        ]
    )


def _pipeline(estimator, scale: bool) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(scale)), ("model", estimator)])


def model_specs() -> dict[str, dict]:
    """Registry of every trainable model plus its search space.

    `tune=False` models are fitted directly; they exist as references.
    """
    return {
        "Linear Regression": {
            "pipeline": _pipeline(LinearRegression(), scale=True),
            "params": {},
            "tune": False,
        },
        "KNN": {
            "pipeline": _pipeline(KNeighborsRegressor(n_jobs=1), scale=True),
            "params": {
                "model__n_neighbors": [5, 10, 20, 30, 50],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            "tune": True,
        },
        "Random Forest": {
            "pipeline": _pipeline(
                RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=1),
                scale=False,
            ),
            "params": {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 12, 20, 30],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
            "tune": True,
        },
        "HistGradientBoosting": {
            "pipeline": _pipeline(
                HistGradientBoostingRegressor(random_state=config.RANDOM_STATE),
                scale=False,
            ),
            "params": {
                "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "model__max_iter": [200, 400, 800],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__min_samples_leaf": [10, 20, 50],
                "model__l2_regularization": [0.0, 0.1, 1.0],
            },
            "tune": True,
        },
    }


def fit_model(
    name: str,
    spec: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[object, dict]:
    """Fit one model, tuning it with time-aware cross-validation if asked."""
    cv = TimeSeriesSplit(n_splits=config.N_CV_SPLITS)

    if not spec["tune"]:
        pipe = spec["pipeline"].fit(X_train, y_train)
        cv_rmse = -_cv_score(pipe, X_train, y_train, cv)
        print(f"[train] {name:22s} fitted")
        return pipe, {"cv_rmse": cv_rmse, "best_params": {}}

    search = RandomizedSearchCV(
        estimator=spec["pipeline"],
        param_distributions=spec["params"],
        n_iter=config.N_SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    print(
        f"[train] {name:22s} cv RMSE {(-search.best_score_):7.2f}   "
        f"params {search.best_params_ or '-'}"
    )
    return search.best_estimator_, {
        "cv_rmse": -search.best_score_,
        "best_params": search.best_params_,
    }


def _cv_score(model, X, y, cv) -> float:
    """Mean RMSE across the time-series folds for an already-fitted model."""
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1
    )
    return float(scores.mean())


def persistence_predictions(X_test: pd.DataFrame) -> pd.Series:
    """Baseline: next hour's volume is assumed equal to the current hour's."""
    return X_test["lag_0h"].copy()


def feature_importance(model, top_n: int = 15) -> pd.DataFrame:
    """Readable feature importances for tree models."""
    estimator = model.named_steps["model"]
    preprocessor = model.named_steps["prep"]

    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    names = preprocessor.get_feature_names_out()
    names = [n.split("__", 1)[-1] for n in names]
    return (
        pd.DataFrame({"feature": names, "importance": estimator.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
