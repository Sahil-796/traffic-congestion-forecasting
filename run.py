"""Entry point - runs the complete pipeline in order.

    python run.py

Stages map to the assignment brief:
    1  load + clean + validate        -> data.py
    2  exploratory data analysis      -> eda.py
    3  feature engineering + split    -> features.py
    4  baselines + model training     -> models.py
    5  tuning (inside fit_model)      -> models.py
    6  evaluation, comparison, errors -> evaluate.py
    7  demonstration on unseen input  -> this file
"""
from __future__ import annotations

import os

# Must happen before numpy/BLAS is imported. Without this every joblib worker
# inherits a full BLAS thread pool, so N workers x 14 threads oversubscribes
# the CPU and the machine stalls.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import json

import joblib
import pandas as pd

import config
import data
import eda
import evaluate
import features
import models


def _stringify(params: dict) -> str:
    if not params:
        return "-"
    return ", ".join(f"{k.replace('model__', '')}={v}" for k, v in params.items())


def main() -> None:
    # ---------------------------------------------------------------- data
    print("\n=== 1. DATA ===")
    df = data.get_clean()

    # ----------------------------------------------------------------- EDA
    print("\n=== 2. EDA ===")
    eda.run_eda(df)

    # ------------------------------------------------------------ features
    print("\n=== 3. FEATURES + SPLIT ===")
    X, y = features.make_dataset(df)
    X_train, X_test, y_train, y_test = features.time_split(X, y)

    # --------------------------------------------------------------- train
    print("\n=== 4. MODELS ===")
    results: list[dict] = []
    fitted: dict[str, tuple[object, object]] = {}

    # Baseline A: persistence. No fitting - it is just "repeat the last value".
    persistence = models.persistence_predictions(X_test)
    results.append(
        {
            "Model": "Persistence (baseline)",
            **evaluate.regression_metrics(y_test, persistence),
            "CV_RMSE": None,
            "Best_Params": "-",
        }
    )
    print(f"[baseline] Persistence            MAE {results[-1]['MAE']:.2f}")

    for name, spec in models.model_specs().items():
        model, info = models.fit_model(name, spec, X_train, y_train)
        pred = model.predict(X_test)
        metrics = evaluate.regression_metrics(y_test, pred)

        results.append(
            {
                "Model": name,
                **metrics,
                "CV_RMSE": round(info["cv_rmse"], 2),
                "Best_Params": _stringify(info["best_params"]),
            }
        )
        fitted[name] = (model, pred)
        print(f"[test]     {name:22s} MAE {metrics['MAE']:7.2f}   "
              f"RMSE {metrics['RMSE']:7.2f}   R2 {metrics['R2']:.4f}")

    # ------------------------------------------------------------- compare
    print("\n=== 5. COMPARISON ===")
    table = evaluate.comparison_table(results)
    table.to_csv(config.OUT_TAB / "model_comparison.csv", index=False)
    print(table.to_string(index=False))

    # The final model is chosen on cross-validated error, NOT on test error.
    tunable = [r for r in results if r["CV_RMSE"] is not None]
    best_row = min(tunable, key=lambda r: r["CV_RMSE"])
    best_name = best_row["Model"]
    best_model, best_pred = fitted[best_name]
    print(f"\n[select] best by CV RMSE: {best_name}")

    joblib.dump(best_model, config.OUT_MODEL / "best_model.joblib", compress=3)

    # ------------------------------------------------------ error analysis
    print("\n=== 6. ERROR ANALYSIS ===")
    evaluate.plot_actual_vs_predicted(y_test, best_pred, best_name)
    evaluate.plot_residuals(y_test, best_pred, best_name)
    by_hour = evaluate.plot_error_by_hour(y_test, best_pred, best_name)
    by_weather = evaluate.plot_error_by_weather(
        y_test, best_pred, X_test["weather_main"], best_name
    )
    by_hour.to_frame("MAE").to_csv(config.OUT_TAB / "error_by_hour.csv")
    by_weather.to_csv(config.OUT_TAB / "error_by_weather.csv")

    worst = evaluate.worst_predictions(y_test, best_pred, X_test)
    worst.to_csv(config.OUT_TAB / "worst_predictions.csv")
    print("\nworst 10 predictions:")
    print(worst.head(10).to_string())

    # ------------------------------------------------------- interpretation
    rf_model, _ = fitted["Random Forest"]
    importance = models.feature_importance(rf_model)
    importance.to_csv(config.OUT_TAB / "feature_importance.csv", index=False)
    evaluate.plot_feature_importance(importance)
    print("\ntop features (Random Forest):")
    print(importance.head(10).to_string(index=False))

    # --------------------------------------------------------- demonstration
    print("\n=== 7. DEMONSTRATION (unseen rows) ===")
    idx = X_test.index
    samples = {
        "weekday rush hour (17:00)": idx[(idx.day_of_week < 5) & (idx.hour == 17)][0],
        "weekend midday (Sun 12:00)": idx[(idx.day_of_week == 6) & (idx.hour == 12)][0],
        "weekday overnight (02:00)": idx[(idx.day_of_week < 5) & (idx.hour == 2)][0],
    }

    demo_rows = []
    for label, ts in samples.items():
        row = X_test.loc[[ts]]
        predicted = float(best_model.predict(row)[0])
        actual = float(y_test.loc[ts])
        print(f"\n{label}  [{ts:%Y-%m-%d %H:%M}]")
        print(f"  weather            : {row['weather_main'].iloc[0]}")
        print(f"  volume this hour   : {row['lag_0h'].iloc[0]:.0f}")
        print(f"  predicted next hour: {predicted:.0f}  -> {features.congestion_tier(predicted)}")
        print(f"  actual next hour   : {actual:.0f}  -> {features.congestion_tier(actual)}")
        print(f"  absolute error     : {abs(predicted - actual):.0f} vehicles/hour")
        demo_rows.append(
            {
                "timestamp": ts.isoformat(),
                "scenario": label,
                "volume_now": float(row["lag_0h"].iloc[0]),
                "predicted": round(predicted, 1),
                "predicted_tier": features.congestion_tier(predicted),
                "actual": actual,
                "actual_tier": features.congestion_tier(actual),
                "abs_error": round(abs(predicted - actual), 1),
            }
        )
    pd.DataFrame(demo_rows).to_csv(config.OUT_TAB / "demonstration.csv", index=False)

    # ------------------------------------------------------------- persist
    summary = {
        "dataset": config.DATASET_NAME,
        "rows_raw_after_clean": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "best_model_by_cv": best_name,
        "metrics": {
            r["Model"]: {
                "MAE": round(r["MAE"], 2),
                "RMSE": round(r["RMSE"], 2),
                "R2": round(r["R2"], 4),
                "CV_RMSE": r["CV_RMSE"],
            }
            for r in results
        },
    }
    (config.OUT_TAB / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== DONE ===")
    print("figures -> outputs/figures")
    print("tables  -> outputs/tables")
    print("model   -> outputs/models/best_model.joblib")


if __name__ == "__main__":
    main()
