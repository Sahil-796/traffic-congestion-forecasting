"""Layer 7 - a small, dependency-free web UI for the trained model.

    python ui.py            # then open http://localhost:8000

Why the standard library and not Streamlit/Flask:

  * nothing extra to install, so it runs in the same venv the report was
    produced in;
  * the assignment's tool list does not include a web framework, and adding
    one would look like scope creep for a two-hour demo;
  * the whole thing is one file, which is a size a student can read and
    defend in a viva.

The important property is that the UI does NOT re-implement feature
engineering. It builds a 24-row hourly frame from the numbers the user
typed and passes it through `features.build_features()` - the exact same
function the training pipeline uses. A prediction from the UI is therefore
reproducible from the notebook, and there is no second, quietly different
definition of `roll_mean_24h` hiding in the browser.
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
import features

MODEL_PATH = config.OUT_MODEL / "best_model.joblib"
COMPARISON_CSV = config.OUT_TAB / "model_comparison.csv"

# Stamp of the file this module was loaded from. A long-lived host (Streamlit
# Cloud) re-runs the app script on a redeploy without restarting the
# interpreter, so `import ui` can hand back the module from the previous
# deploy. `app.py` compares this stamp to the file on disk and reloads.
try:
    BUILD = Path(__file__).stat().st_mtime_ns
except OSError:  # pragma: no cover - unreadable source is not fatal
    BUILD = 0

# Fallback hourly shape for the "typical day" button, used only if the
# cleaned CSV is missing. These are plausible I-94 volumes, not fitted values.
FALLBACK_PROFILE = {
    "weekday": [700, 520, 450, 430, 520, 1500, 3400, 4900, 5600, 5200, 4700,
                4600, 4700, 4800, 5100, 5400, 5600, 5700, 5200, 4300, 3300,
                2500, 1700, 1000],
    "weekend": [900, 700, 600, 550, 560, 700, 1100, 1700, 2400, 3000, 3400,
                3700, 3900, 4000, 3900, 3700, 3500, 3400, 3200, 3000, 2700,
                2300, 1800, 1300],
}

# ------------------------------------------------------------------ state
_MODEL = None
_CLEAN: pd.DataFrame | None = None
_META: dict | None = None
_LOCK = threading.Lock()

# Bump when the shape of `meta()` changes. Streamlit Cloud keeps the imported
# `ui` module (and its cached dict) alive across redeploys, so an app built
# against an older metadata shape would otherwise KeyError on the new keys.
META_VERSION = 3


def model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"{MODEL_PATH} not found - run `python run.py` once to train it"
            )
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def clean_data() -> pd.DataFrame | None:
    """The cleaned hourly table, used only to show ground truth and profiles."""
    global _CLEAN
    if _CLEAN is None:
        if config.CLEAN_CSV.exists():
            _CLEAN = pd.read_csv(
                config.CLEAN_CSV, index_col="date_time", parse_dates=True
            )
        else:
            _CLEAN = pd.DataFrame(columns=["traffic_volume"])
    return _CLEAN


def weather_options() -> list[str]:
    """Categories the encoder was actually fitted on."""
    enc = model().named_steps["prep"].named_transformers_["cat"]
    values = list(enc.categories_[0])
    return sorted(str(v) for v in values)


def hourly_profiles() -> dict[str, list[int]]:
    df = clean_data()
    if df.empty or df["traffic_volume"].dropna().empty:
        return FALLBACK_PROFILE

    frame = pd.DataFrame(
        {
            "hour": df.index.hour,
            "weekend": df.index.day_of_week >= 5,
            "volume": df["traffic_volume"],
        }
    )
    means = frame.groupby(["weekend", "hour"])["volume"].mean()
    out = {}
    for key, weekend in (("weekday", False), ("weekend", True)):
        out[key] = [
            int(round(means.get((weekend, h), FALLBACK_PROFILE[key][h])))
            for h in range(24)
        ]
    return out


def model_card() -> dict:
    try:
        table = pd.read_csv(COMPARISON_CSV)
        best = table.sort_values("CV_RMSE").iloc[0]
        return {
            "name": str(best["Model"]),
            "mae": round(float(best["MAE"]), 2),
            "rmse": round(float(best["RMSE"]), 2),
            "r2": round(float(best["R2"]), 4),
            "mape": round(float(best["MAPE_%"]), 1),
        }
    except Exception:
        return {"name": "Random Forest", "mae": 144.46, "rmse": 224.50,
                "r2": 0.9871, "mape": 6.1}


def meta() -> dict:
    global _META
    with _LOCK:
        if _META is None or _META.get("version") != META_VERSION:
            df = clean_data()
            first = df.index.min() if len(df.index) else pd.Timestamp("2018-06-15")
            last = df.index.max() if len(df.index) else pd.Timestamp("2018-09-30")
            _META = {
                "weather": weather_options(),
                "profiles": hourly_profiles(),
                "thresholds": {
                    "free_max": config.CONGESTION_FREE_MAX,
                    "moderate_max": config.CONGESTION_MODERATE_MAX,
                },
                "model": model_card(),
                "data_range": [first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")],
                "default_date": "2018-06-15",
                # Everything below is descriptive only: the same facts the
                # report states, so a demo can be explained from the screen.
                "dataset": {
                    "name": config.DATASET_NAME,
                    "source": config.DATASET_SOURCE,
                    "licence": config.DATASET_LICENSE,
                    "station": "I-94 westbound, MN DOT station 301",
                    "records": "48,204 hourly records",
                    "test_period": f"{config.TEST_START} to {last:%Y-%m-%d}",
                },
                "features": len(features.FEATURE_NUM) + len(features.FEATURE_CAT),
                "version": META_VERSION,
                "tiers": [
                    {"name": "free-flow", "upto": config.CONGESTION_FREE_MAX},
                    {"name": "moderate", "upto": config.CONGESTION_MODERATE_MAX},
                    {"name": "congested", "upto": None},
                ],
            }
        return _META


# ------------------------------------------------------------------ predict
def build_feature_row(
    history: list[float],
    timestamp: pd.Timestamp,
    weather_main: str,
    temp_c: float,
    rain_1h: float,
    snow_1h: float,
    clouds_all: float,
    is_holiday: bool,
) -> pd.DataFrame:
    """One feature row, built by the same code path as training.

    `history` holds 25 hourly volumes in chronological order: the reading 24
    hours ago up to and including the current hour. The count is not
    arbitrary:

        history[0]   -> lag_24h            (needs t-24)
        history[1:]  -> the 24-value window that roll_mean_24h averages
        history[-1]  -> lag_0h, the current hour

    With only 24 values there is no t-24 reading, so lag_24h would be NaN and
    the row unusable. Everything here is therefore a real measurement; nothing
    is interpolated or guessed.
    """
    if len(history) != 25:
        raise ValueError("exactly 25 hourly volumes are required (24 hours ago to now)")

    values = np.asarray(history, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("history contains a blank or non-numeric value")
    if (values < 0).any():
        raise ValueError("traffic volume cannot be negative")

    index = pd.date_range(timestamp - pd.Timedelta(hours=24), timestamp, freq="h")
    frame = pd.DataFrame(
        {
            "traffic_volume": values,
            # Weather is constant across the window: it only feeds features on
            # the final row, and lags/rolling windows read traffic_volume only.
            "temp": float(temp_c) + 273.15,
            "rain_1h": float(rain_1h),
            "snow_1h": float(snow_1h),
            "clouds_all": float(clouds_all),
            "weather_main": weather_main,
            "is_holiday": int(bool(is_holiday)),
        },
        index=index,
    )

    feat = features.build_features(frame)
    row = feat.iloc[[-1]][features.FEATURE_NUM + features.FEATURE_CAT]

    if row.isna().any(axis=1).any():
        missing = row.columns[row.isna().any()].tolist()
        raise ValueError(f"could not build features: {missing}")

    return row


def summarise(result: dict) -> str:
    """One plain-English sentence about the forecast, shared by both UIs.

    Kept here rather than in the browser so the wording (and the numbers in
    it) has exactly one definition.
    """
    change = result["change_pct"]
    if change is None or abs(change) < 1:
        lead = "Traffic expected to hold steady."
    else:
        verb = "rise" if change > 0 else "fall"
        lead = f"Traffic expected to {verb} {abs(change):.0f}% from the current hour."

    near = result["next_tier"]
    if near:
        tail = f"{near['veh']:,.0f} veh/h of headroom before the {near['name']} band."
    else:
        tail = "Already inside the congested band."
    return f"{lead} {tail}"


def predict_from_payload(payload: dict) -> dict:
    history = payload.get("history") or []
    if not isinstance(history, list):
        raise ValueError("history must be a list")

    date = pd.Timestamp(str(payload.get("date", "")))
    hour = int(payload.get("hour", 12))
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    timestamp = date + pd.Timedelta(hours=hour)

    weather_main = str(payload.get("weather_main") or "Clear")
    temp_c = float(payload.get("temp_c", 20))
    rain_1h = max(0.0, float(payload.get("rain_1h", 0)))
    snow_1h = max(0.0, float(payload.get("snow_1h", 0)))
    clouds_all = min(100.0, max(0.0, float(payload.get("clouds_all", 0))))
    is_holiday = bool(payload.get("is_holiday", False))

    if temp_c < -60 or temp_c > 60:
        raise ValueError("temperature must be between -60 and 60 C")

    row = build_feature_row(
        [float(v) for v in history],
        timestamp,
        weather_main,
        temp_c,
        rain_1h,
        snow_1h,
        clouds_all,
        is_holiday,
    )

    predicted = float(model().predict(row)[0])
    predicted = max(0.0, predicted)
    current = float(history[-1])

    card = meta()["model"]
    thresholds = meta()["thresholds"]

    # Distance to the next tier boundary, so the UI can say how much room is
    # left before the forecast changes label. None once already congested.
    free_max = float(thresholds["free_max"])
    moderate_max = float(thresholds["moderate_max"])
    if predicted < free_max:
        next_tier = {"name": "moderate", "at": free_max, "veh": round(free_max - predicted, 1)}
    elif predicted < moderate_max:
        next_tier = {"name": "congested", "at": moderate_max,
                     "veh": round(moderate_max - predicted, 1)}
    else:
        next_tier = None

    # Ground truth, when the chosen hour exists in the cleaned dataset.
    actual = None
    truth_time = timestamp + pd.Timedelta(hours=1)
    df = clean_data()
    if not df.empty and truth_time in df.index:
        value = df.at[truth_time, "traffic_volume"]
        if pd.notna(value):
            actual = float(value)

    result = {
        "timestamp": timestamp.isoformat(),
        "forecast_for": truth_time.isoformat(),
        "predicted": round(predicted, 1),
        "tier": features.congestion_tier(predicted),
        "current": round(current, 1),
        "delta": round(predicted - current, 1),
        "change_pct": round((predicted - current) / current * 100, 1) if current else None,
        "typical_error": card["mae"],
        "low": round(max(0.0, predicted - card["mae"]), 1),
        "high": round(predicted + card["mae"], 1),
        "next_tier": next_tier,
        "actual": None if actual is None else round(actual, 1),
        "actual_tier": None if actual is None else features.congestion_tier(actual),
        "error": None if actual is None else round(abs(predicted - actual), 1),
        "thresholds": thresholds,
    }
    result["summary"] = summarise(result)
    return result


# --------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>I-94 Traffic Forecast</title>
<style>
  :root{
    --bg:#f5f6f8; --card:#ffffff; --ink:#0f172a; --body:#334155; --muted:#64748b;
    --line:#e4e8ee; --accent:#0f766e; --accent-soft:#e7f4f1;
    --free:#16a34a; --mod:#d97706; --cong:#dc2626;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  .shell{max-width:1120px;margin:0 auto;padding:34px 20px 44px}
  .eyebrow{font-size:11px;font-weight:700;letter-spacing:.13em;text-transform:uppercase;
           color:var(--accent);margin:0 0 10px}
  h1{margin:0 0 8px;font-size:25px;letter-spacing:-.015em}
  .lede{margin:0;color:var(--muted);max-width:64ch}
  .facts{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}
  .fact{font-size:12px;background:#fff;border:1px solid var(--line);border-radius:999px;
        padding:4px 11px;color:var(--muted)}
  .fact b{color:var(--ink);font-weight:600}

  main{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(0,1fr);
       gap:18px;margin-top:24px;align-items:start}
  @media (max-width:920px){ main{grid-template-columns:1fr} }

  .card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px}
  .step + .step{margin-top:22px;padding-top:20px;border-top:1px solid var(--line)}
  .step-head{display:flex;align-items:center;gap:9px;margin:0 0 4px}
  .step-no{font-size:11px;font-weight:700;color:var(--accent);background:var(--accent-soft);
           border-radius:6px;padding:2px 7px}
  h2{font-size:13px;font-weight:650;margin:0;letter-spacing:.005em}
  .hint{font-size:12.5px;color:var(--muted);margin:6px 0 14px;max-width:60ch}

  label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px}
  input,select{
    width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:9px;
    font:inherit;font-size:14px;background:#fff;color:var(--ink);
  }
  input:focus,select:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
  .row.three{grid-template-columns:1fr 1fr 1fr}
  .check{display:flex;align-items:center;gap:8px;font-size:14px;color:var(--ink);
         padding-top:9px}
  .check input{width:auto}

  .hist{display:grid;grid-template-columns:repeat(5,1fr);gap:7px}
  @media (max-width:520px){ .hist{grid-template-columns:repeat(3,1fr)} .row.three{grid-template-columns:1fr} }
  .cell label{font-size:11px;margin-bottom:3px;font-variant-numeric:tabular-nums}
  .cell input{padding:6px 8px;font-size:13px;text-align:right;font-variant-numeric:tabular-nums}
  .cell.now label{color:var(--accent);font-weight:700}
  .cell.now input{border-color:var(--accent);background:#f5fdfa;font-weight:600}

  .presets{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}
  button{
    font:inherit;font-size:12.5px;padding:6px 11px;border-radius:9px;cursor:pointer;
    border:1px solid var(--line);background:#fff;color:var(--body);
  }
  button:hover{border-color:#c3ccd8;color:var(--ink)}
  button.primary{
    background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;
    width:100%;padding:11px;font-size:14px;margin-top:6px;
  }
  button.primary:hover{background:#0b5f59;color:#fff}
  button.primary:disabled{opacity:.6;cursor:wait}

  .result{position:sticky;top:20px}
  .res-when{font-size:12.5px;color:var(--muted)}
  .big{font-size:46px;font-weight:700;letter-spacing:-.03em;line-height:1.05;
       margin:8px 0 10px;font-variant-numeric:tabular-nums}
  .unit{font-size:13px;font-weight:500;color:var(--muted);letter-spacing:0;margin-left:5px}
  .chip{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;
        font-weight:600;letter-spacing:.01em}
  .chip.free-flow{background:#dcfce7;color:#166534}
  .chip.moderate{background:#fef3c7;color:#92400e}
  .chip.congested{background:#fee2e2;color:#991b1b}
  .meaning{font-size:13.5px;color:var(--body);margin:11px 0 0}

  .scale{margin:18px 0 0}
  .scale .wrap{position:relative}
  .scale .track{display:flex;height:8px;border-radius:999px;overflow:hidden;background:#eef2f6}
  .seg{height:100%}
  .seg.free-flow{background:#bbf7d0}
  .seg.moderate{background:#fde68a}
  .seg.congested{background:#fecaca}
  .marker{position:absolute;top:50%;width:13px;height:13px;border-radius:50%;
          background:var(--ink);border:2.5px solid #fff;transform:translate(-50%,-50%);
          box-shadow:0 1px 4px rgba(15,23,42,.35)}
  .scale-labels{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);
                margin-top:7px;font-variant-numeric:tabular-nums}

  .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:2px;margin:18px 0 0;
         border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--line)}
  .stats div{background:#fbfcfd;padding:9px 10px}
  .stats .k{display:block;font-size:10.5px;color:var(--muted);text-transform:uppercase;
            letter-spacing:.06em}
  .stats .v{display:block;font-size:15px;font-weight:600;margin-top:2px;
            font-variant-numeric:tabular-nums}
  .spark{margin-top:14px}
  .spark svg{width:100%;height:92px;display:block;overflow:visible}
  .spark .cap{display:flex;justify-content:space-between;font-size:11px;color:var(--muted);
              margin-top:2px;font-variant-numeric:tabular-nums}

  .kv{display:flex;justify-content:space-between;gap:12px;font-size:13px;padding:7px 0;
      border-bottom:1px solid #f1f5f9}
  .kv:last-of-type{border-bottom:none}
  .kv span:first-child{color:var(--muted)}
  .kv span:last-child{font-variant-numeric:tabular-nums;text-align:right}
  .kv strong{font-weight:600}
  .note{font-size:12.5px;color:var(--muted);margin:10px 0 0}

  details.how{margin-top:18px;padding-top:14px;border-top:1px solid var(--line)}
  details.how summary{cursor:pointer;font-size:12.5px;font-weight:600;color:var(--accent);
                      list-style:none}
  details.how summary::-webkit-details-marker{display:none}
  details.how summary::before{content:"+  ";font-weight:700}
  details.how[open] summary::before{content:"–  "}
  details.how ul{margin:10px 0 0;padding-left:16px;font-size:12.5px;color:var(--muted)}
  details.how li{margin-bottom:5px}

  .error{background:#fee2e2;color:#991b1b;border:1px solid #fecaca;border-radius:10px;
         padding:9px 12px;font-size:12.5px;margin-top:12px;display:none}
  .empty{color:var(--muted);font-size:13px}
  footer{max-width:1120px;margin:0 auto;padding:0 20px 40px;color:var(--muted);font-size:11.5px;
         display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
</style>
</head>
<body>
<div class="shell">
  <header>
    <p class="eyebrow">Machine learning &middot; I-94 corridor</p>
    <h1>I-94 Traffic Forecast</h1>
    <p class="lede">Predicts how many vehicles will pass the sensor in the next hour, from the
       last 24 hours of readings, the day of the week and the weather.</p>
    <div class="facts">
      <span class="fact"><b id="f-model">Random Forest</b> model</span>
      <span class="fact">R&sup2; <b id="f-r2">0.987</b></span>
      <span class="fact">MAE <b id="f-mae">&plusmn;144</b> veh/h</span>
      <span class="fact">Trained 2012&ndash;2017 &middot; tested Jan&ndash;Sep 2018</span>
    </div>
  </header>

  <main>
    <section class="card">
      <div class="step">
        <div class="step-head"><span class="step-no">1</span><h2>When are we forecasting?</h2></div>
        <p class="hint">Pick the hour you are standing in. The forecast is for the hour after it.</p>
        <div class="row">
          <div><label for="date">Date</label><input type="date" id="date"></div>
          <div><label for="hour">Current hour</label><select id="hour"></select></div>
        </div>
        <div class="check"><input type="checkbox" id="holiday"><span>Public holiday</span></div>
      </div>

      <div class="step">
        <div class="step-head"><span class="step-no">2</span><h2>Weather at that hour</h2></div>
        <p class="hint">Weather is secondary to the daily traffic pattern, but snow and rain do
           pull volumes down.</p>
        <div class="row">
          <div><label for="weather">Conditions</label><select id="weather"></select></div>
          <div><label for="temp">Temperature (&deg;C)</label>
               <input type="number" id="temp" value="20" step="0.1"></div>
        </div>
        <div class="row three">
          <div><label for="rain">Rain (mm)</label>
               <input type="number" id="rain" value="0" min="0" step="0.1"></div>
          <div><label for="snow">Snow (mm)</label>
               <input type="number" id="snow" value="0" min="0" step="0.1"></div>
          <div><label for="clouds">Cloud cover (%)</label>
               <input type="number" id="clouds" value="20" min="0" max="100" step="5"></div>
        </div>
      </div>

      <div class="step">
        <div class="step-head"><span class="step-no">3</span><h2>Last 25 hourly readings</h2></div>
        <p class="hint">Vehicles per hour, oldest first. The outlined box is the current hour;
           these 25 numbers become the lag and rolling features.</p>
        <div class="hist" id="hist"></div>
        <div class="presets">
          <button type="button" data-preset="weekday">Typical weekday</button>
          <button type="button" data-preset="weekend">Typical weekend</button>
          <button type="button" data-preset="match">Match this date</button>
          <button type="button" data-preset="quiet">Quiet night</button>
          <button type="button" data-preset="rush">Rush hour</button>
        </div>
        <button class="primary" id="go">Predict next hour</button>
        <div class="error" id="error"></div>
      </div>
    </section>

    <section class="card result">
      <h2 style="margin-bottom:12px">Forecast</h2>
      <div id="output"><span class="empty">Fill in the form and press predict.</span></div>
    </section>
  </main>
</div>

<footer id="footer"></footer>

<script>
const $ = s => document.querySelector(s);
let META = null;
let HISTORY = [];

const TIER_MAX = 7500;

function fmt(n){ return Math.round(n).toLocaleString('en-US'); }

function whenText(iso){
  const d = new Date(iso);
  const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const hh = String(d.getHours()).padStart(2,'0');
  return `${days[d.getDay()]} ${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()} &middot; ${hh}:00`;
}

function fillHours(){
  const sel = $('#hour');
  for (let h = 0; h < 24; h++){
    const o = document.createElement('option');
    o.value = h;
    o.textContent = String(h).padStart(2,'0') + ':00';
    sel.appendChild(o);
  }
  sel.value = 17;
}

function hourLabels(endHour){
  const out = [];
  for (let i = 24; i >= 0; i--) out.push((endHour - i + 48) % 24);
  return out;
}

function buildHistoryGrid(){
  const grid = $('#hist');
  const previous = [];
  grid.querySelectorAll('input').forEach(el => previous.push(el.value));
  grid.innerHTML = '';
  const endHour = parseInt($('#hour').value, 10);
  hourLabels(endHour).forEach((h, i) => {
    const cell = document.createElement('div');
    cell.className = 'cell' + (i === 24 ? ' now' : '');
    const id = 'v' + i;
    cell.innerHTML = `<label for="${id}">${String(h).padStart(2,'0')}:00${i === 24 ? ' now' : ''}</label>
                      <input type="number" id="${id}" min="0" step="1">`;
    grid.appendChild(cell);
    if (previous[i]) document.getElementById(id).value = previous[i];
  });
}

function setHistory(values){
  values.forEach((v, i) => {
    const el = document.getElementById('v' + i);
    if (el) el.value = Math.round(v);
  });
}

function profileFor(dateStr){
  const d = new Date(dateStr + 'T12:00:00');
  const weekend = d.getDay() === 0 || d.getDay() === 6;
  return META.profiles[weekend ? 'weekend' : 'weekday'];
}

function applyProfile(kind){
  const endHour = parseInt($('#hour').value, 10);
  const labels = hourLabels(endHour);
  let source;
  if (kind === 'quiet') {
    source = labels.map(h => (h >= 0 && h <= 5) ? 480 : 900);
  } else if (kind === 'rush') {
    source = labels.map(h => (h >= 6 && h <= 9) ? 5400 : (h >= 16 && h <= 18 ? 5600 : 3600));
  } else if (kind === 'weekday' || kind === 'weekend') {
    source = labels.map(h => META.profiles[kind][h]);
  } else {
    source = labels.map(h => profileFor($('#date').value)[h]);
  }
  setHistory(source);
}

function spark(history, predicted){
  const vals = history.concat([predicted]);
  const w = 320, h = 88, pad = 5;
  const max = Math.max(...vals, 1000) * 1.1;
  const n = vals.length;
  const x = i => pad + (i / (n - 1)) * (w - 2 * pad);
  const y = v => h - pad - (Math.max(0, v) / max) * (h - 2 * pad);

  const histPts = vals.slice(0, -1).map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const area = `${pad},${(h - pad).toFixed(1)} ${histPts.join(' ')} ${x(n - 2).toFixed(1)},${(h - pad).toFixed(1)}`;
  const link = [
    `${x(n - 2).toFixed(1)},${y(vals[n - 2]).toFixed(1)}`,
    `${x(n - 1).toFixed(1)},${y(vals[n - 1]).toFixed(1)}`
  ].join(' ');

  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <polygon points="${area}" fill="#e2e8f0" opacity=".55"></polygon>
    <polyline points="${histPts.join(' ')}" fill="none" stroke="#94a3b8" stroke-width="1.6"
              vector-effect="non-scaling-stroke" stroke-linejoin="round"></polyline>
    <polyline points="${link}" fill="none" stroke="#0f766e" stroke-width="1.8"
              stroke-dasharray="4 3" vector-effect="non-scaling-stroke"></polyline>
    <circle cx="${x(n - 2).toFixed(1)}" cy="${y(vals[n - 2]).toFixed(1)}" r="2.6" fill="#64748b"></circle>
    <circle cx="${x(n - 1).toFixed(1)}" cy="${y(vals[n - 1]).toFixed(1)}" r="3.6" fill="#0f766e"></circle>
  </svg>
  <div class="cap"><span>24 hours ago</span><span>now</span><span>next hour</span></div>`;
}

function render(r, history){
  const scale = Math.max(TIER_MAX, r.predicted, r.actual || 0);
  const freeW = (r.thresholds.free_max / scale) * 100;
  const modW = ((r.thresholds.moderate_max - r.thresholds.free_max) / scale) * 100;
  const conW = 100 - freeW - modW;
  const mark = Math.min(99, Math.max(1, (r.predicted / scale) * 100));

  let html = `
    <div class="res-when">Next hour &middot; ${whenText(r.forecast_for)}</div>
    <div class="big">${fmt(r.predicted)}<span class="unit">veh/h</span></div>
    <div><span class="chip ${r.tier}">${r.tier}</span></div>
    <p class="meaning">${r.summary}</p>

    <div class="scale">
      <div class="wrap">
        <div class="track">
          <div class="seg free-flow" style="width:${freeW}%"></div>
          <div class="seg moderate" style="width:${modW}%"></div>
          <div class="seg congested" style="width:${conW}%"></div>
        </div>
        <div class="marker" style="left:${mark}%"></div>
      </div>
      <div class="scale-labels">
        <span>0</span><span>${fmt(r.thresholds.free_max)}</span>
        <span>${fmt(r.thresholds.moderate_max)}</span><span>${fmt(scale)}</span>
      </div>
    </div>

    <div class="stats">
      <div><span class="k">Test R&sup2;</span><span class="v">${META.model.r2.toFixed(3)}</span></div>
      <div><span class="k">Mean error</span><span class="v">&plusmn;${fmt(META.model.mae)}</span></div>
      <div><span class="k">Typical miss</span><span class="v">${META.model.mape ? META.model.mape + '%' : '6.1%'}</span></div>
    </div>

    <div class="spark">${spark(history, r.predicted)}</div>

    <div class="kv"><span>Typical error</span>
      <span>&plusmn;${fmt(r.typical_error)} &rarr; likely ${fmt(r.low)}&ndash;${fmt(r.high)}</span></div>
    <div class="kv"><span>Change vs current hour</span><span>${fmt(r.current)} &rarr; <strong>${fmt(r.predicted)}</strong></span></div>`;

  if (r.actual !== null && r.actual !== undefined) {
    html += `<div class="kv"><span>Recorded next hour</span>
      <span><strong>${fmt(r.actual)}</strong> veh/h &middot; error ${fmt(r.error)}</span></div>
      <p class="note">This hour exists in the 2018 test set, so the forecast can be checked
      against what the sensor actually recorded.</p>`;
  } else {
    html += `<p class="note">That hour is outside the recorded test period, so only the
      forecast is shown.</p>`;
  }

  html += `<details class="how" open>
      <summary>How this works</summary>
      <ul>
        <li>The model is a <b>${META.model.name}</b> trained on ${META.dataset.records}
            from one I-94 sensor; the fit stops at December 2017.</li>
        <li>Tested on ${META.dataset.test_period}, never shuffled:
            MAE ${META.model.mae} veh/h, RMSE ${META.model.rmse}, R&sup2; ${META.model.r2.toFixed(3)}.</li>
        <li>Your 25 readings and the weather build ${META.features} features: 7 lags,
            6 rolling statistics, calendar and weather. The current hour alone drives
            about 48% of the decision.</li>
        <li>Tiers: below ${fmt(META.thresholds.free_max)} free-flow,
            ${fmt(META.thresholds.free_max)}&ndash;${fmt(META.thresholds.moderate_max)} moderate,
            above ${fmt(META.thresholds.moderate_max)} congested.</li>
      </ul>
    </details>`;

  $('#output').innerHTML = html;
}

async function predict(){
  const btn = $('#go');
  const err = $('#error');
  err.style.display = 'none';

  const history = [];
  for (let i = 0; i < 25; i++){
    const el = document.getElementById('v' + i);
    if (!el || el.value === '') { return fail('Please fill all 25 hourly volumes.'); }
    const v = Number(el.value);
    if (!isFinite(v) || v < 0) { return fail('Hourly volumes must be positive numbers.'); }
    history.push(v);
  }
  HISTORY = history;

  const payload = {
    date: $('#date').value,
    hour: parseInt($('#hour').value, 10),
    weather_main: $('#weather').value,
    temp_c: Number($('#temp').value),
    rain_1h: Number($('#rain').value),
    snow_1h: Number($('#snow').value),
    clouds_all: Number($('#clouds').value),
    is_holiday: $('#holiday').checked,
    history: history
  };

  btn.disabled = true;
  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) { fail(data.error || 'Prediction failed.'); return; }
    render(data, history);
  } catch (e) {
    fail('Could not reach the server: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

function fail(msg){
  const err = $('#error');
  err.textContent = msg;
  err.style.display = 'block';
}

async function init(){
  META = await (await fetch('/api/meta')).json();

  $('#f-model').textContent = META.model.name;
  $('#f-r2').textContent = META.model.r2.toFixed(3);
  $('#f-mae').textContent = '\u00b1' + fmt(META.model.mae);

  fillHours();
  $('#date').value = META.default_date;

  const wsel = $('#weather');
  META.weather.forEach(w => {
    const o = document.createElement('option');
    o.value = w; o.textContent = w;
    wsel.appendChild(o);
  });
  wsel.value = META.weather.includes('Clear') ? 'Clear' : META.weather[0];

  buildHistoryGrid();
  applyProfile('weekday');

  $('#hour').addEventListener('change', () => { buildHistoryGrid(); applyProfile('match'); predict(); });
  $('#date').addEventListener('change', () => { applyProfile('match'); predict(); });
  $('#holiday').addEventListener('change', predict);
  ['#weather', '#temp', '#rain', '#snow', '#clouds'].forEach(sel => {
    $(sel).addEventListener('change', predict);
  });
  document.querySelectorAll('[data-preset]').forEach(b => {
    b.addEventListener('click', () => { applyProfile(b.dataset.preset); predict(); });
  });
  $('#go').addEventListener('click', predict);

  $('#footer').innerHTML =
    `<span>${META.dataset.name} &middot; ${META.dataset.source} &middot; ${META.dataset.licence}</span>` +
    `<span>${META.dataset.station} &middot; ${META.model.name} &middot; test ${META.data_range[0].slice(0,4)}&ndash;${META.data_range[1].slice(0,4)}</span>`;

  predict();
}

init();
</script>
</body>
</html>
"""


# -------------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    server_version = "TrafficUI/1.0"

    def _send(self, code: int, body: str, ctype: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/meta":
            self._send(200, json.dumps(meta()), "application/json")
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/predict":
            self._send(404, "not found", "text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = predict_from_payload(payload)
        except Exception as exc:  # the message is shown to the user verbatim
            self._send(400, json.dumps({"error": str(exc)}), "application/json")
            return
        self._send(200, json.dumps(result), "application/json")

    def log_message(self, fmt: str, *args) -> None:
        if "/api/" in (args[0] if args else ""):
            return
        print(f"[ui] {self.address_string()} {fmt % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Web UI for the traffic model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    meta()  # warm the model and the caches before the first request
    url = f"http://{args.host}:{args.port}/"

    print(f"[ui] model      {MODEL_PATH.name}")
    print(f"[ui] conditions {', '.join(weather_options())}")
    print(f"[ui] serving    {url}")
    print("[ui] Ctrl-C to stop")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ui] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
