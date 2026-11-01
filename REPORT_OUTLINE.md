# Report Outline

**ECSCI24302 – Machine Learning Essentials, Project-Based Learning (PBL)**
**Project: Traffic Congestion Forecasting Using Machine Learning**

This document is a skeleton for the final report and follows the required 15-section structure.

---

## Title Page

- Project title, course code and name, student name and ID, submission date.
- Declaration of original work and acknowledgement of the dataset donor (John Hogue, UCI Machine Learning Repository).

## Abstract / Project Summary

- One-paragraph statement of the task: one-hour-ahead forecasting of hourly traffic volume on I-94 westbound, Minnesota, framed as supervised regression.
- Summary of the approach: lag, rolling, calendar and weather features; chronological split; two baselines and three candidate models with randomised hyperparameter search.
- Headline finding in qualitative terms, with final MAE, RMSE and R² figures inserted once the pipeline has been run.

## 1. Introduction

- Context: recurrent congestion on urban interstate corridors and the value of short-horizon volume forecasts for operations and traveller information.
- Scope: a single sensor on I-94 westbound between Minneapolis and St Paul, hourly observations from October 2012 to September 2018.
- Contribution: a fully reproducible pipeline comparing classical baselines against tree-based and instance-based learners under a strictly temporal evaluation protocol.

## 2. Problem Statement and Motivation

- Formal statement: given observations up to hour *t*, predict `traffic_volume` at hour *t+1*; target defined as `LEAD(traffic_volume, 1)`.
- Motivation for one-hour horizon: short enough to be actionable for ramp metering and traveller alerts, long enough to be learnable from lagged volume alone.
- Why regression rather than classification: volume is continuous and ordinal; congestion tiers are derived downstream from predicted volume (<1500 free-flow, 1500–4000 moderate, >4000 congested).
- Why temporal ordering matters: shuffling would leak future information and inflate apparent performance.

## 3. Dataset Description and Source

- Provenance: UCI Machine Learning Repository id 492, "Metro Interstate Traffic Volume"; DOI 10.24432/C5X60B; CC BY 4.0; John Hogue, 2019.
- Scale and granularity: 48,204 raw rows, 9 columns, hourly, October 2012 – September 2018; single ATR station 301, I-94 westbound, Minnesota.
- Variables: `holiday`, `temp` (K), `rain_1h` (mm), `snow_1h` (mm), `clouds_all` (%), `weather_main` (11 values), `weather_description` (38 values), `date_time`, `traffic_volume` (0–7280 vehicles/hour).
- Justification for use: open licence, credible government source, and a realistic set of imperfections suitable for demonstrating preprocessing decisions.

## 4. Data Understanding and EDA

- Target distribution: shape, skew and multimodality; the twin weekday rush-hour peaks versus the flatter weekend profile.
- Temporal structure: hourly, daily and weekly seasonality; annual trend; autocorrelation of `traffic_volume` at lags 1–24, justifying the lag feature set.
- Weather relationships: volume against temperature, rain, snow and `weather_main`; note that adverse weather effects are modest relative to the commute cycle.
- Data-quality diagnostics: 7,629 duplicate timestamps, 17 exact duplicate rows, 11,976 of 52,551 expected hourly slots absent, `temp` = 0.0 K on 10 rows, `rain_1h` maximum 9,831.3 mm, `holiday` blank on 48,143 rows, `traffic_volume` = 0 on 2 rows.

## 5. Data Preprocessing

- Duplicate collapsing: aggregate the 7,629 duplicate timestamps to one row per hour (mean for continuous measures, mode for categorical), then drop the 17 exact duplicate rows.
- Gap handling: reindex to a complete hourly grid; document that 11,976 of 52,551 slots (22.8%) are absent, and state the chosen policy (drop leading/trailing gaps, do not impute the target, restrict lag construction to contiguous runs).
- Invalid value removal: drop or null the 10 rows with `temp` = 0.0 K as physically impossible.
- Outlier treatment: cap `rain_1h` at a plausible physical bound given the 9,831.3 mm maximum, and treat the 2 zero-volume rows as suspect.
- Categorical handling: recode blank `holiday` to "not a holiday" before deriving a binary flag; one-hot encode `weather_main`.

## 6. Feature Engineering and Feature Selection

- Lag features at 0, 1, 2, 3, 6, 12 and 24 hours, chosen to span the short-term and daily cycles. Lag 0 is the row's own reading, which is known at prediction time and is expected to dominate importance.
- Rolling statistics (mean and standard deviation) over 3, 6 and 24 hours to capture local level and volatility.
- Calendar features: hour of day, day of week, month, `is_weekend`, `is_rush_hour`.
- Weather features: temperature converted from Kelvin to Celsius, `rain_1h`, `snow_1h`, `clouds_all`, one-hot `weather_main`; rationale for excluding the 38-level `weather_description`.
- Leakage control: all rolling and lag features computed strictly from past values; target shifted with `LEAD` and the final row of each contiguous run discarded.

## 7. Machine Learning Methodology

- Chronological split: train October 2012 – December 2017, test January – September 2018, never shuffled.
- Validation: `TimeSeriesSplit` with 5 splits, applied to the training data only, so that every validation fold lies strictly after its training fold.
- Model families: an instance-based learner (KNN), a bagged tree ensemble (Random Forest) and a boosted tree ensemble (HistGradientBoosting), each compared against two baselines.
- Metrics: MAE, RMSE and R², reported on validation and on the held-out test period; RMSE emphasised because large errors on peak hours matter operationally.

## 8. Baseline Model

- Persistence baseline: next hour equals the current hour, requiring no fitting — establishes the floor any learned model must beat.
- Linear Regression baseline: ordinary least squares on the full engineered feature set, providing a parametric reference and an interpretable coefficient view.
- Purpose: to distinguish genuine learning from the trivial predictability of autocorrelated volume.

## 9. Model Development

- KNN Regressor: distance-based benchmark sensitive to feature scaling; requires standardisation of continuous features.
- Random Forest: non-parametric ensemble handling non-linearity and interactions without scaling; discussion of depth and estimator count.
- HistGradientBoostingRegressor: histogram-based boosting with native handling of missing values and efficient training on the hourly dataset.
- Shared pipeline discipline: identical features, identical split indices and identical random seed across all models.

## 10. Hyperparameter Tuning

- Search procedure: `RandomizedSearchCV`, 30 iterations, 5-fold `TimeSeriesSplit`, scoring `neg_root_mean_squared_error`.
- Search spaces: number of neighbours and weighting for KNN; number of trees, maximum depth and minimum samples per leaf for Random Forest; learning rate, maximum iterations, maximum leaf nodes and L2 regularisation for HistGradientBoosting.
- Rationale for randomised rather than exhaustive search: cost control with a high-dimensional space and a strictly sequential validation scheme.
- Reporting: record best parameters and best validation RMSE per model for the comparison table.

## 11. Results and Model Comparison

- Present the comparison table (below) with validation and test metrics for all five estimators.
- Narrative ordering: persistence floor, linear reference, then the three learned models; quantify the improvement of the best model over persistence in MAE and RMSE terms.
- Test-period behaviour: January – September 2018 only, with commentary on whether validation ranking transfers to the hold-out period.

### Model Comparison Table

| Model | Best Hyperparameters | Validation Result | Test Metric(s) | Observations |
|---|---|---|---|---|
| Persistence (baseline) | — (no fitting) | TBD | MAE: TBD; RMSE: TBD; R²: TBD | — |
| Linear Regression (baseline) | — (no tuning) | TBD | MAE: TBD; RMSE: TBD; R²: TBD | — |
| KNN | TBD | TBD | MAE: TBD; RMSE: TBD; R²: TBD | — |
| Random Forest | TBD | TBD | MAE: TBD; RMSE: TBD; R²: TBD | — |
| HistGradientBoosting | TBD | TBD | MAE: TBD; RMSE: TBD; R²: TBD | — |

## 12. Error Analysis

- Residual distribution overall and by congestion tier (<1500, 1500–4000, >4000): whether errors are larger in absolute terms at high volume but comparable in relative terms.
- Temporal error structure: concentration of large errors around transition hours, public holidays and adverse-weather events.
- Worst-case inspection: the largest absolute errors and the conditions under which they occur.
- Seasonal and split-boundary effects: behaviour in the early months of the 2018 test period, and whether performance drifts across the test window.

## 13. Model Interpretation

- Feature importance from the tree ensembles, expected to be dominated by the one-hour lag and the 24-hour lag/rolling features.
- Calendar effects: consistency of the learned rush-hour and weekend patterns with the EDA findings.
- Weather contribution: relative importance of temperature and precipitation once lag structure is accounted for.
- Operational reading: translation of predicted volumes into the three congestion tiers and what a one-tier misclassification would mean in practice.

## 14. Limitations

- Single sensor only: the data carry no spatial information, so upstream and downstream propagation of congestion cannot be represented or exploited.
- Volume is a proxy for congestion rather than a direct congestion measurement; no speed, occupancy or travel-time data are available, so flow can be high while still free-moving.
- 22.8% of expected hourly slots are missing (11,976 of 52,551), which constrains lag construction and leaves the series irregular.
- Coverage is uneven over time: 2014–2015 are poorly covered, so seasonal and annual patterns are learned from an unbalanced sample.
- Results are not directly comparable to graph-neural-network benchmarks on METR-LA, which use a multi-sensor network, a different road system and a different target definition.
- The one-hour-ahead task is intrinsically favourable: consecutive hourly readings are strongly autocorrelated and the current hour's volume is a legitimate input, so high R² values reflect task difficulty rather than a superior model. A multi-hour horizon, where the latest observation is unavailable, would be a materially harder problem.
- The holiday flag is extremely rare (61 flagged hours out of 48,204), so its estimated effect is unstable and should not be over-interpreted.

## 15. Conclusion and Future Work

- Restate the task and the headline comparison: whether learned models materially beat persistence, and by how much.
- Summarise methodological lessons: the necessity of a chronological split, the value of explicit baselines, and the sensitivity of results to gap and outlier handling.
- Future work: multi-sensor or loop-detector fusion, incorporation of speed and occupancy to measure congestion directly, probabilistic forecasting intervals, comparison against sequence models (LSTM, temporal CNN) and spatial models on a richer network, and enrichment with incident and roadwork records.

## References

- Hogue, J. (2019) *Metro Interstate Traffic Volume*. UCI Machine Learning Repository. DOI: 10.24432/C5X60B. Licence: CC BY 4.0.
- Scikit-learn documentation for `TimeSeriesSplit`, `RandomizedSearchCV` and `HistGradientBoostingRegressor`.
- Additional methodological references on time-series cross-validation and traffic forecasting, cited in the report body.
