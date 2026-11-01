"""Layer 4 - metrics, comparison table and error analysis."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

sns.set_theme(style="whitegrid", context="notebook")
FIG_DPI = 150


# ------------------------------------------------------------------ metrics
def regression_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape}


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["MAE"] = df["MAE"].round(2)
    df["RMSE"] = df["RMSE"].round(2)
    df["R2"] = df["R2"].round(4)
    df["MAPE_%"] = df["MAPE_%"].round(2)
    return df.sort_values("MAE").reset_index(drop=True)


# ------------------------------------------------------------------- figures
def plot_actual_vs_predicted(y_true, y_pred, model_name: str, n: int = 500):
    """Plot the first `n` test points, breaking the line at missing hours.

    Without the reindex, matplotlib joins the point before a gap directly to
    the point after it, drawing a straight diagonal that looks like model
    output when it is really absent data.
    """
    idx = y_true.index[: min(n, len(y_true))]
    grid = pd.date_range(idx.min(), idx.max(), freq="h")

    actual = y_true.reindex(grid)
    predicted = pd.Series(np.asarray(y_pred, float), index=y_true.index).reindex(grid)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(grid, actual.values, label="actual", lw=1.4, color="#1f2937")
    ax.plot(grid, predicted.values, label="predicted", lw=1.4, alpha=0.85, color="#dc2626")
    ax.set_title(f"Actual vs predicted - {model_name} (first {len(idx)} test hours)")
    ax.set_ylabel("vehicles / hour")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "actual_vs_predicted.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name: str):
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(resid, bins=60, ax=axes[0], color="#2563eb")
    axes[0].set_title(f"Residual distribution - {model_name}")
    axes[0].set_xlabel("actual - predicted (vehicles/hour)")

    sns.boxplot(x=resid, ax=axes[1], color="#93c5fd")
    axes[1].set_title("Residual spread")
    axes[1].set_xlabel("actual - predicted (vehicles/hour)")

    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "residuals.png", dpi=FIG_DPI)
    plt.close(fig)


def plot_error_by_hour(y_true, y_pred, model_name: str):
    err = pd.DataFrame(
        {
            "hour": y_true.index.hour,
            "abs_error": np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)),
        }
    )
    grouped = err.groupby("hour")["abs_error"].mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=grouped.index, y=grouped.values, ax=ax, color="#0ea5e9")
    ax.set_title(f"Mean absolute error by hour of day - {model_name}")
    ax.set_xlabel("hour")
    ax.set_ylabel("MAE (vehicles/hour)")
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "error_by_hour.png", dpi=FIG_DPI)
    plt.close(fig)
    return grouped


def plot_error_by_weather(y_true, y_pred, weather: pd.Series, model_name: str):
    err = pd.DataFrame(
        {
            "weather": weather.reindex(y_true.index).values,
            "abs_error": np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)),
        }
    ).dropna()

    grouped = err.groupby("weather")["abs_error"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 50].sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=grouped.index, y=grouped["mean"], ax=ax, color="#10b981")
    ax.set_title(f"Mean absolute error by weather condition - {model_name}")
    ax.set_ylabel("MAE (vehicles/hour)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "error_by_weather.png", dpi=FIG_DPI)
    plt.close(fig)
    return grouped


def plot_feature_importance(importance: pd.DataFrame):
    if importance.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=importance, y="feature", x="importance", ax=ax, color="#6366f1"
    )
    ax.set_title("Random Forest feature importance")
    ax.set_xlabel("relative importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / "feature_importance.png", dpi=FIG_DPI)
    plt.close(fig)


def worst_predictions(y_true, y_pred, X_test: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "actual": y_true.values,
            "predicted": np.asarray(y_pred, float),
            "abs_error": np.abs(y_true.values - np.asarray(y_pred, float)),
        },
        index=y_true.index,
    )
    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.day_of_week
    out["weather"] = X_test["weather_main"].reindex(out.index).values
    return out.sort_values("abs_error", ascending=False).head(n).round(1)
