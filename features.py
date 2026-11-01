"""Layer 2 - feature engineering, target construction and the time split.

Two rules are enforced here and nowhere else:

  * every feature is built from information available *at or before* the
    row's timestamp (lags and rolling windows only look backwards);
  * the training/test boundary is a date, never a random split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Hours of history used as raw lag features.
# lag_0h is the reading for the row's own timestamp. Predicting t+1 from t
# means that value is known at prediction time, so it must be included -
# it is the single strongest signal in the dataset.
LAGS = (0, 1, 2, 3, 6, 12, 24)
# Trailing windows used for the rolling summaries.
WINDOWS = (3, 6, 24)

FEATURE_NUM = (
    [f"lag_{h}h" for h in LAGS]
    + [f for w in WINDOWS for f in (f"roll_mean_{w}h", f"roll_std_{w}h")]
    + [
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "is_rush",
        "temp_c",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "is_holiday",
    ]
)
FEATURE_CAT = ["weather_main"]
TARGET = "target_volume_next"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lag, rolling, calendar and weather features plus the target."""
    volume = df["traffic_volume"]
    out = pd.DataFrame(index=df.index)

    # --- past traffic -------------------------------------------------
    for h in LAGS:
        out[f"lag_{h}h"] = volume.shift(h)

    for w in WINDOWS:
        # min_periods=w keeps the window honest: a gap anywhere inside the
        # window produces NaN instead of a silently wrong average.
        out[f"roll_mean_{w}h"] = volume.rolling(w, min_periods=w).mean()
        out[f"roll_std_{w}h"] = volume.rolling(w, min_periods=w).std()

    # --- calendar -----------------------------------------------------
    idx = df.index
    out["hour"] = idx.hour
    out["dayofweek"] = idx.day_of_week
    out["month"] = idx.month
    out["is_weekend"] = (idx.day_of_week >= 5).astype("int8")

    rush_hours = ((idx.hour >= 7) & (idx.hour <= 9)) | (
        (idx.hour >= 16) & (idx.hour <= 18)
    )
    out["is_rush"] = (rush_hours & (idx.day_of_week < 5)).astype("int8")

    # --- weather ------------------------------------------------------
    out["temp_c"] = df["temp"] - 273.15
    out["rain_1h"] = df["rain_1h"]
    out["snow_1h"] = df["snow_1h"]
    out["clouds_all"] = df["clouds_all"]
    out["weather_main"] = df["weather_main"]
    out["is_holiday"] = df["is_holiday"]

    # --- target: LEAD(volume, 1) -------------------------------------
    out[TARGET] = volume.shift(-1)

    return out


def make_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and target with incomplete rows removed."""
    feat = build_features(df)

    complete = (
        feat[FEATURE_NUM + FEATURE_CAT].notna().all(axis=1)
        & feat[TARGET].notna()
    )

    print(f"[features] rows before filtering : {len(feat):,}")
    print(f"[features] rows dropped (no history/target): {len(feat) - complete.sum():,}")
    print(f"[features] usable rows           : {complete.sum():,}")

    X = feat.loc[complete, FEATURE_NUM + FEATURE_CAT]
    y = feat.loc[complete, TARGET]
    return X, y


def time_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split. The test period is never shuffled into training."""
    cutoff = pd.Timestamp(config.TEST_START)
    train = X.index < cutoff
    test = ~train

    X_train, y_train = X.loc[train], y.loc[train]
    X_test, y_test = X.loc[test], y.loc[test]

    # Hard guard: if this ever fires, the split logic has leaked the future
    # into training and every downstream number is meaningless.
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("empty train or test split")
    if X_train.index.max() >= X_test.index.min():
        raise ValueError("data leakage: training data extends past the test boundary")

    print(
        f"[split] train {X_train.index.min():%Y-%m-%d} -> {X_train.index.max():%Y-%m-%d}"
        f"  ({len(X_train):,} rows)"
    )
    print(
        f"[split] test  {X_test.index.min():%Y-%m-%d} -> {X_test.index.max():%Y-%m-%d}"
        f"  ({len(X_test):,} rows)"
    )
    return X_train, X_test, y_train, y_test


def congestion_tier(volume: float | np.ndarray):
    """Map predicted volume (vehicles/hour) to an operational label."""
    if isinstance(volume, pd.Series):
        return pd.cut(
            volume,
            bins=[-np.inf, config.CONGESTION_FREE_MAX, config.CONGESTION_MODERATE_MAX, np.inf],
            labels=["free-flow", "moderate", "congested"],
        )
    if volume < config.CONGESTION_FREE_MAX:
        return "free-flow"
    if volume < config.CONGESTION_MODERATE_MAX:
        return "moderate"
    return "congested"
