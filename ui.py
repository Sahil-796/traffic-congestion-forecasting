"""Layer 7 - a small, dependency-free web UI for the trained model.

    python ui.py            # then open http://localhost:8000

Why the standard library and not Streamlit/Flask:

  * nothing extra to install, so it runs in the same venv the report was
    produced in;
  * the assignment's tool list does not include a web framework, and adding
    one would look like scope creep for a two-hour demo;
  * the whole thing is one file and roughly 400 lines, which is a size a
    student can read and defend in a viva.

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

import joblib
import numpy as np
import pandas as pd

import config
import features

MODEL_PATH = config.OUT_MODEL / "best_model.joblib"
COMPARISON_CSV = config.OUT_TAB / "model_comparison.csv"

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
        }
    except Exception:
        return {"name": "Random Forest", "mae": 144.46, "rmse": 224.50, "r2": 0.9871}


def meta() -> dict:
    global _META
    with _LOCK:
        if _META is None:
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

    # Ground truth, when the chosen hour exists in the cleaned dataset.
    actual = None
    truth_time = timestamp + pd.Timedelta(hours=1)
    df = clean_data()
    if not df.empty and truth_time in df.index:
        value = df.at[truth_time, "traffic_volume"]
        if pd.notna(value):
            actual = float(value)

    return {
        "timestamp": timestamp.isoformat(),
        "predicted": round(predicted, 1),
        "tier": features.congestion_tier(predicted),
        "current": round(current, 1),
        "delta": round(predicted - current, 1),
        "typical_error": card["mae"],
        "low": round(max(0.0, predicted - card["mae"]), 1),
        "high": round(predicted + card["mae"], 1),
        "actual": None if actual is None else round(actual, 1),
        "actual_tier": None if actual is None else features.congestion_tier(actual),
        "error": None if actual is None else round(abs(predicted - actual), 1),
        "thresholds": meta()["thresholds"],
    }


# --------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>I-94 Traffic Forecast</title>
<style>
  :root{
    --bg:#f4f5f7; --card:#ffffff; --ink:#111827; --muted:#6b7280;
    --line:#e5e7eb; --accent:#0f766e; --accent-soft:#ccfbf1;
    --free:#16a34a; --mod:#d97706; --cong:#dc2626;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  header{max-width:1100px;margin:0 auto;padding:34px 20px 18px}
  header h1{margin:0 0 6px;font-size:22px;letter-spacing:-.01em}
  header p{margin:0;color:var(--muted);max-width:60ch}
  main{
    max-width:1100px;margin:0 auto;padding:0 20px 60px;
    display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:20px;
    align-items:start;
  }
  @media (max-width:900px){ main{grid-template-columns:1fr} }
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
  .fieldset{margin-bottom:20px}
  .fieldset:last-child{margin-bottom:0}
  h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
     margin:0 0 12px;font-weight:600}
  label{display:block;font-size:13px;color:var(--muted);margin-bottom:5px}
  input,select{
    width:100%;padding:8px 10px;border:1px solid var(--line);border-radius:8px;
    font:inherit;font-size:14px;background:#fff;color:var(--ink);
  }
  input:focus,select:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
  .row.three{grid-template-columns:1fr 1fr 1fr}
  .row.one{grid-template-columns:1fr}
  .check{display:flex;align-items:center;gap:8px;font-size:14px;color:var(--ink);margin-top:4px}
  .check input{width:auto}
  .hist{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
  @media (max-width:640px){ .hist{grid-template-columns:repeat(4,1fr)} .row.three{grid-template-columns:1fr} }
  .hist .cell label{font-size:11px;margin-bottom:3px;color:var(--muted);font-variant-numeric:tabular-nums}
  .hist input{padding:6px 8px;font-size:13px;text-align:right}
  .hint{font-size:12px;color:var(--muted);margin:0 0 10px}
  .presets{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
  button{
    font:inherit;font-size:13px;padding:7px 12px;border-radius:8px;cursor:pointer;
    border:1px solid var(--line);background:#fff;color:var(--ink);
  }
  button:hover{border-color:#cbd5e1}
  button.primary{
    background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600;
    width:100%;padding:11px;font-size:14px;
  }
  button.primary:disabled{opacity:.6;cursor:wait}
  .result{position:sticky;top:20px}
  .big{font-size:52px;font-weight:700;letter-spacing:-.03em;line-height:1;margin:6px 0 2px;
       font-variant-numeric:tabular-nums}
  .unit{font-size:14px;font-weight:500;color:var(--muted);letter-spacing:0}
  .chip{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;
        font-weight:600;letter-spacing:.02em}
  .chip.free-flow{background:#dcfce7;color:#166534}
  .chip.moderate{background:#fef3c7;color:#92400e}
  .chip.congested{background:#fee2e2;color:#991b1b}
  .band{color:var(--muted);font-size:13px;margin-top:8px}
  .bar{margin:16px 0 4px}
  .bar .track{height:9px;background:#f1f5f9;border-radius:999px;overflow:hidden}
  .bar .fill{height:100%;border-radius:999px;transition:width .35s ease}
  .bar .meta{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);
             margin-top:5px;font-variant-numeric:tabular-nums}
  .divider{height:1px;background:var(--line);margin:18px 0}
  .kv{display:flex;justify-content:space-between;font-size:13px;padding:5px 0}
  .kv span:first-child{color:var(--muted)}
  .kv span:last-child{font-variant-numeric:tabular-nums}
  .actual{margin-top:14px;padding:12px;border-radius:10px;background:#f8fafc;
          border:1px solid var(--line);font-size:13px}
  .actual strong{font-variant-numeric:tabular-nums}
  .error{background:#fee2e2;color:#991b1b;border:1px solid #fecaca;border-radius:10px;
         padding:10px 12px;font-size:13px;margin-top:14px;display:none}
  .empty{color:var(--muted);font-size:13px}
  footer{max-width:1100px;margin:0 auto;padding:0 20px 40px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>I-94 Traffic Forecast</h1>
  <p>Predicts westbound traffic volume for the next hour from the last 24 hours of
     readings, calendar position and weather. Same model, same feature code as the report.</p>
</header>

<main>
  <section class="card">
    <div class="fieldset">
      <h2>When</h2>
      <div class="row one">
        <div><label for="date">Date</label><input type="date" id="date"></div>
      </div>
      <div class="row">
        <div><label for="hour">Hour of day</label><select id="hour"></select></div>
        <div><label for="holiday">Public holiday</label>
             <div class="check"><input type="checkbox" id="holiday"><span>Yes, holiday</span></div>
        </div>
      </div>
    </div>

    <div class="fieldset">
      <h2>Weather at this hour</h2>
      <div class="row">
        <div><label for="weather">Conditions</label><select id="weather"></select></div>
        <div><label for="temp">Temperature (&deg;C)</label><input type="number" id="temp" value="20" step="0.1"></div>
      </div>
      <div class="row three">
        <div><label for="rain">Rain (mm)</label><input type="number" id="rain" value="0" min="0" step="0.01"></div>
        <div><label for="snow">Snow (mm)</label><input type="number" id="snow" value="0" min="0" step="0.01"></div>
        <div><label for="clouds">Cloud cover (%)</label><input type="number" id="clouds" value="20" min="0" max="100" step="1"></div>
      </div>
    </div>

    <div class="fieldset">
      <h2>Traffic readings</h2>
      <p class="hint">Vehicles per hour, oldest first: the value 24 hours ago through to the
         current hour. The last box is the current hour, and it is what the model forecasts
         one hour beyond.</p>
      <div class="hist" id="hist"></div>
      <div class="presets">
        <button type="button" data-preset="weekday">Load typical weekday</button>
        <button type="button" data-preset="weekend">Load typical weekend</button>
        <button type="button" data-preset="quiet">Load quiet night</button>
        <button type="button" data-preset="rush">Load rush hour</button>
      </div>
    </div>

    <div class="fieldset">
      <button class="primary" id="go">Predict next hour</button>
      <div class="error" id="error"></div>
    </div>
  </section>

  <section class="card result">
    <h2>Forecast</h2>
    <div id="output"><span class="empty">Fill in the form and press predict.</span></div>
  </section>
</main>

<footer id="footer"></footer>

<script>
const $ = s => document.querySelector(s);
let META = null;

function fmt(n){ return Math.round(n).toLocaleString('en-US'); }

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
  grid.innerHTML = '';
  const endHour = parseInt($('#hour').value, 10);
  const labels = hourLabels(endHour);
  labels.forEach((h, i) => {
    const cell = document.createElement('div');
    cell.className = 'cell';
    const id = 'v' + i;
    cell.innerHTML = `<label for="${id}">${String(h).padStart(2,'0')}:00</label>
                      <input type="number" id="${id}" min="0" step="1">`;
    grid.appendChild(cell);
  });
}

function setHistory(values){
  values.forEach((v, i) => { const el = document.getElementById('v' + i); if (el) el.value = Math.round(v); });
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
  } else {
    const prof = (kind === 'weekend' || kind === 'weekday') ? META.profiles[kind] : profileFor($('#date').value);
    source = labels.map(h => prof[h]);
  }
  setHistory(source);
}

function render(r){
  const maxScale = Math.max(6000, r.current, r.predicted) * 1.05;
  const cur = Math.max(2, (r.current / maxScale) * 100);
  const pre = Math.max(2, (r.predicted / maxScale) * 100);
  const dir = r.delta > 0 ? 'higher' : (r.delta < 0 ? 'lower' : 'flat');
  const col = r.tier === 'free-flow' ? 'var(--free)' : (r.tier === 'moderate' ? 'var(--mod)' : 'var(--cong)');

  let html = `
    <div class="big">${fmt(r.predicted)}<span class="unit"> veh/h</span></div>
    <div><span class="chip ${r.tier}">${r.tier}</span></div>
    <div class="band">Typical error &plusmn;${fmt(r.typical_error)} &rarr; likely ${fmt(r.low)}&ndash;${fmt(r.high)}</div>

    <div class="bar">
      <div class="track"><div class="fill" style="width:${cur}%;background:#94a3b8"></div></div>
      <div class="meta"><span>now</span><span>${fmt(r.current)}</span></div>
    </div>
    <div class="bar">
      <div class="track"><div class="fill" style="width:${pre}%;background:${col}"></div></div>
      <div class="meta"><span>next hour</span><span>${fmt(r.predicted)}</span></div>
    </div>
    <div class="kv"><span>Change</span><span>${dir === 'flat' ? 'no change' : (r.delta > 0 ? '+' : '') + fmt(r.delta)} veh/h</span></div>
    <div class="kv"><span>Free-flow below</span><span>${fmt(r.thresholds.free_max)}</span></div>
    <div class="kv"><span>Congested above</span><span>${fmt(r.thresholds.moderate_max)}</span></div>
  `;

  if (r.actual !== null && r.actual !== undefined) {
    html += `<div class="actual">
      Recorded next hour: <strong>${fmt(r.actual)}</strong> veh/h
      <span class="chip ${r.actual_tier}" style="margin-left:6px">${r.actual_tier}</span><br>
      Absolute error: <strong>${fmt(r.error)}</strong> veh/h
    </div>`;
  } else {
    html += `<div class="actual">No recorded value for that hour in the dataset, so only the forecast is shown.</div>`;
  }

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
    render(data);
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
  applyProfile(null);

  $('#hour').addEventListener('change', () => { buildHistoryGrid(); applyProfile(null); });
  $('#date').addEventListener('change', () => applyProfile(null));
  document.querySelectorAll('[data-preset]').forEach(b => {
    b.addEventListener('click', () => applyProfile(b.dataset.preset));
  });
  $('#go').addEventListener('click', predict);

  $('#footer').textContent =
    `Model: ${META.model.name} | test MAE ${META.model.mae} veh/h | R2 ${META.model.r2} | ` +
    `dataset covers ${META.data_range[0]} to ${META.data_range[1]}`;
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
