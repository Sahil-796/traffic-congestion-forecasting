# Traffic Congestion Forecasting Using Machine Learning

**ECSCI24302 – Machine Learning Essentials, Project-Based Learning (PBL)**

This project forecasts hourly traffic volume on a Minnesota interstate corridor one hour ahead, using six years of weather and calendar signals from a single road sensor. It frames congestion forecasting as a supervised regression task, compares two baselines against three machine learning models, and evaluates them on a strictly chronological hold-out period. All results are produced by a single reproducible pipeline.

## Dataset

**Metro Interstate Traffic Volume** — UCI Machine Learning Repository, dataset id 492.

| Attribute | Value |
|---|---|
| Source | <https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip> |
| DOI | [10.24432/C5X60B](https://doi.org/10.24432/C5X60B) |
| Licence | CC BY 4.0 |
| Authors / donor | John Hogue, 2019 |
| Size | 48,204 raw rows × 9 columns, hourly |
| Period | October 2012 – September 2018 |
| Sensor | I-94 westbound ATR station 301, between Minneapolis and St Paul, Minnesota |

**Columns:** `holiday`, `temp` (Kelvin), `rain_1h` (mm), `snow_1h` (mm), `clouds_all` (%), `weather_main` (11 values), `weather_description` (38 values), `date_time`, `traffic_volume` (target, integer, 0–7280 vehicles/hour).

**Known data-quality issues** (quantified during profiling):

- 7,629 duplicate timestamps and 17 exact duplicate rows
- 11,976 of 52,551 expected hourly slots absent
- `temp` = 0.0 K on 10 rows (physically invalid)
- `rain_1h` maximum 9,831.3 mm (physically impossible)
- `holiday` blank on 48,143 rows (blank means "not a holiday")
- `traffic_volume` = 0 on 2 rows

## Problem Statement

Given observations up to hour *t*, predict `traffic_volume` at hour *t+1*. This is a supervised regression problem with a temporal ordering that must be respected throughout. The target is constructed as `LEAD(traffic_volume, 1)` over time.

Features comprise lag features (0, 1, 2, 3, 6, 12, 24 hours, where lag 0 is the row's own reading), rolling mean and standard deviation over 3, 6 and 24 hours, calendar features (hour, day of week, month, `is_weekend`, `is_rush_hour`), weather variables (temperature in Celsius, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`) and a holiday flag.

Predicted volumes are additionally mapped to congestion tiers: **< 1500 free-flow**, **1500–4000 moderate**, **> 4000 congested**.

## Repository Structure

```
.
├── config.py         # paths, constants, thresholds, random seed (42)
├── data.py           # download, clean, validate → one tidy hourly table
├── features.py       # feature engineering, target, chronological split
├── models.py         # model pipelines and hyperparameter search spaces
├── evaluate.py       # metrics, comparison table, error analysis, figures
├── eda.py            # exploratory data analysis plots
├── ui.py             # small web UI for manual predictions (stdlib only)
├── app.py            # Streamlit front end, same model and feature code
├── run.py            # single entry point for the whole pipeline
├── requirements.txt
├── data/
│   ├── raw/          # downloaded archive, untouched
│   └── processed/    # cleaned, feature-engineered tables
└── outputs/
    ├── figures/
    ├── tables/
    └── models/
```

## Setup

Requires Python 3.13.

```bash
uv venv
uv pip install -r requirements.txt
```

## Running

```bash
python run.py
```

This executes the full pipeline in order: download and clean the raw data, build features and the chronological split, run EDA, fit baselines and candidate models, tune hyperparameters, and evaluate.

## Interactive UI

Two front ends ship with the project. They share `ui.predict_from_payload()`, so the same inputs always produce the same number in both.

### Streamlit

```bash
streamlit run app.py
```

Opens <http://localhost:8501>. Widgets for date, hour, weather and holiday, plus an editable 25-row table of hourly readings with presets. The forecast updates as you change anything; no predict button.

### Standard library

```bash
python ui.py
```

Opens <http://localhost:8000>. No framework, no extra dependency.

### What the UI asks for, and why 25 readings

The page takes a date and hour, the weather at that hour, and the **25 hourly readings from 24 hours ago up to and including the current hour**. The count is not arbitrary:

```
history[0]    = 24 h ago  → lag_24h
history[1:25] = last 24 h → roll_mean_24h, roll_std_24h
history[24]   = now       → lag_0h
```

With 24 values there is no reading from `t-24`, so `lag_24h` would be missing and the row unusable. Both UIs build the row by calling `features.build_features()`, so a prediction made in the browser reproduces the pipeline exactly.

| Option | Default | Meaning |
|---|---|---|
| `--port` | `8000` / `8501` | Port to listen on |
| `--host` | `127.0.0.1` | Interface to bind (`ui.py` only) |
| `--no-browser` | off | Do not open a browser on start (`ui.py` only) |

### Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. On <https://share.streamlit.io>, choose **New app**, pick the repo, branch `main`, and set the main file to `app.py`.
3. Nothing else — `outputs/models/best_model.joblib` (65 MB) is committed so the app can predict immediately.

The model is the only large file in the repository. It is tracked deliberately: it is what the hosted app loads at startup, and retraining on every deploy would take eight minutes. `.gitignore` keeps every other `.joblib` out.

If you would rather not carry 65 MB, `app.py` still starts without it and shows a clear "model not available" message instead of a traceback.


## Outputs

| Location | Contents |
|---|---|
| `data/raw/` | Original downloaded archive, preserved unmodified |
| `data/processed/` | Cleaned hourly table and feature matrices |
| `outputs/figures/` | EDA plots, prediction-vs-actual charts, residual plots |
| `outputs/tables/` | Metric comparison table, error analysis tables |
| `outputs/models/` | Serialised fitted models |

## Methodology

Five estimators are compared on identical features and splits:

1. **Persistence baseline** — next hour equals this hour
2. **Linear Regression** — baseline parametric reference
3. **K-Nearest Neighbours Regressor**
4. **Random Forest**
5. **HistGradientBoostingRegressor**

Hyperparameters are selected with `RandomizedSearchCV` (20 iterations, 5-fold `TimeSeriesSplit`, scoring `neg_root_mean_squared_error`) on the training data only. Models are assessed using MAE, RMSE and R².

## Reproducibility

- All randomness is governed by a fixed seed of **42**, set in `config.py`.
- The train/test split is **chronological and never shuffled**: training covers October 2012 – December 2017, testing covers January – September 2018. This prevents future information leaking into the past.
- Cross-validation uses `sklearn.model_selection.TimeSeriesSplit` with 5 splits, applied to the training data only.
- Running `python run.py` from a clean checkout regenerates every table and figure in `outputs/`.
