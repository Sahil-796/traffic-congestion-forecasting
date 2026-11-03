"""Streamlit front end for the traffic model.

    streamlit run app.py

Deployment alternative to `ui.py`. Both call the same functions - the model is
loaded once, the feature row is built by `features.build_features()`, and
`ui.predict_from_payload()` does the prediction - so the two UIs cannot drift
apart or disagree about a number.

Streamlit reloads this script on every widget interaction, which is why the
readings grid is kept in `st.session_state`: otherwise the user's edits would
be wiped by the rerun that their own keystroke triggered.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import ui

st.set_page_config(
    page_title="I-94 Traffic Forecast",
    page_icon="🚗",
    layout="wide",
)

TIER_STYLE = {
    "free-flow": "🟢",
    "moderate": "🟠",
    "congested": "🔴",
}

PRESETS = ["Weekday", "Weekend", "This date", "Quiet night", "Rush hour"]

# Small amount of CSS so the Streamlit page and `ui.py` share one visual
# language. Everything else is native Streamlit widgets.
CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1180px;}
  .fact {display:inline-block; font-size:12px; background:#fff; border:1px solid #e4e8ee;
         border-radius:999px; padding:3px 10px; margin:0 6px 0 0; color:#64748b;}
  .fact b {color:#0f172a; font-weight:600;}
  .tier-scale {margin:14px 0 4px;}
  .tier-track {display:flex; height:8px; border-radius:999px; overflow:hidden;
               background:#eef2f6;}
  .tier-track div {height:100%;}
  .tier-wrap {position:relative;}
  .tier-marker {position:absolute; top:50%; width:13px; height:13px; border-radius:50%;
                background:#0f172a; border:2.5px solid #fff; transform:translate(-50%,-50%);
                box-shadow:0 1px 4px rgba(15,23,42,.35);}
  .tier-labels {display:flex; justify-content:space-between; font-size:11px;
                color:#64748b; margin-top:6px; font-variant-numeric:tabular-nums;}
  .stat-strip {display:grid; grid-template-columns:repeat(3,1fr); gap:1px;
               background:#e4e8ee; border:1px solid #e4e8ee; border-radius:10px;
               overflow:hidden; margin:14px 0 6px;}
  .stat-strip div {background:#fbfcfd; padding:9px 10px;}
  .stat-strip .k {display:block; font-size:10.5px; text-transform:uppercase;
                  letter-spacing:.06em; color:#64748b;}
  .stat-strip .v {display:block; font-size:15px; font-weight:600; margin-top:2px;
                  font-variant-numeric:tabular-nums;}
  .kv {display:flex; justify-content:space-between; font-size:13px; padding:6px 0;
       border-bottom:1px solid #f1f5f9;}
  .kv span:first-child {color:#64748b;}
  .kv span:last-child {font-variant-numeric:tabular-nums;}
  .meaning {font-size:14px; color:#334155; margin:10px 0 0;}
</style>
"""


@st.cache_resource(show_spinner="Loading model ...")
def boot() -> dict:
    """Load the model and the descriptive metadata exactly once per process."""
    meta = ui.meta()
    return {"meta": meta, "weather": ui.weather_options()}


def fill(kind: str, hour: int, date) -> list[int]:
    """25 volumes ending at `hour`, from a preset or the hourly means."""
    meta = boot()["meta"]
    labels = [(hour - i) % 24 for i in range(24, -1, -1)]

    if kind == "Quiet night":
        return [480 if 0 <= h <= 5 else 900 for h in labels]
    if kind == "Rush hour":
        return [5400 if 6 <= h <= 9 else (5600 if 16 <= h <= 18 else 3600) for h in labels]

    weekend = kind == "Weekend" or (kind == "This date" and date.weekday() >= 5)
    profile = meta["profiles"]["weekend" if weekend else "weekday"]
    return [int(profile[h]) for h in labels]


def refresh(sig: tuple, kind: str, hour: int, date) -> None:
    """Regenerate the grid only when the shape of the question has changed.

    The editor widget keeps its own state, so its key is dropped as well.
    Without that, a preset would update `readings` while the visible table
    kept showing the old numbers.
    """
    if st.session_state.get("sig") != sig:
        st.session_state.readings = fill(kind, hour, date)
        st.session_state.sig = sig
        st.session_state.pop("grid", None)


def tier_scale(meta: dict, predicted: float) -> str:
    """Three-segment congestion bar with a marker at the forecast."""
    free_max = meta["thresholds"]["free_max"]
    moderate_max = meta["thresholds"]["moderate_max"]
    top = max(7500, predicted * 1.05)
    free_w = free_max / top * 100
    mod_w = (moderate_max - free_max) / top * 100
    con_w = 100 - free_w - mod_w
    mark = min(99.0, max(1.0, predicted / top * 100))
    return f"""
<div class="tier-scale">
  <div class="tier-wrap">
    <div class="tier-track">
      <div style="width:{free_w}%;background:#bbf7d0"></div>
      <div style="width:{mod_w}%;background:#fde68a"></div>
      <div style="width:{con_w}%;background:#fecaca"></div>
    </div>
    <div class="tier-marker" style="left:{mark}%"></div>
  </div>
  <div class="tier-labels">
    <span>0</span><span>{free_max:,.0f}</span><span>{moderate_max:,.0f}</span>
    <span>{top:,.0f}</span>
  </div>
</div>"""


def history_chart(readings: list[int], predicted: float) -> alt.Chart:
    """The 25 readings with the forecast appended as a dashed continuation."""
    rows = [{"step": i, "veh/h": v, "series": "last 24 hours"}
            for i, v in enumerate(readings)]
    rows += [{"step": 24, "veh/h": readings[-1], "series": "forecast"},
             {"step": 25, "veh/h": predicted, "series": "forecast"}]
    frame = pd.DataFrame(rows)

    base = alt.Chart(frame).encode(
        x=alt.X(
            "step:Q", title=None,
            axis=alt.Axis(
                values=[0, 12, 25],
                labelExpr="{0:'-24 h', 12:'-12 h', 25:'next hour'}[datum.value]",
            ),
        ),
        y=alt.Y("veh/h:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color(
            "series:N", legend=None,
            scale=alt.Scale(domain=["last 24 hours", "forecast"],
                            range=["#94a3b8", "#0f766e"]),
        ),
    )
    line = base.mark_line(strokeWidth=2).encode(
        strokeDash=alt.StrokeDash(
            "series:N", legend=None,
            scale=alt.Scale(domain=["last 24 hours", "forecast"], range=[[1, 0], [4, 3]]),
        )
    )
    end = base.mark_point(size=70, filled=True).transform_filter(
        alt.datum.step == 25
    )
    now = alt.Chart(pd.DataFrame({"step": [24]})).mark_rule(
        strokeDash=[3, 3], color="#cbd5e1"
    ).encode(x="step:Q")
    return (line + end + now).properties(height=170)


def main() -> None:
    try:
        loaded = boot()
    except RuntimeError as exc:
        st.error(
            f"**Model not available.**  \n{exc}  \n"
            "A fresh clone has no trained model, because `outputs/models/` is "
            "git-ignored. Run `python run.py` once to build it (about 8 minutes)."
        )
        st.stop()

    meta = loaded["meta"]
    weather_options = loaded["weather"]
    st.markdown(CSS, unsafe_allow_html=True)

    st.title("I-94 Traffic Forecast")
    st.caption(
        "Predicts how many vehicles will pass the sensor in the next hour, from the last "
        "24 hours of readings, the day of the week and the weather."
    )
    st.markdown(
        f'<span class="fact"><b>{meta["model"]["name"]}</b> model</span>'
        f'<span class="fact">R² <b>{meta["model"]["r2"]:.3f}</b></span>'
        f'<span class="fact">MAE <b>±{meta["model"]["mae"]:,.0f}</b> veh/h</span>'
        f'<span class="fact">Trained 2012–2017 · tested Jan–Sep 2018</span>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown("**1 · When are we forecasting?**")
        st.caption("Pick the hour you are standing in. The forecast is for the hour after it.")
        c1, c2, c3 = st.columns([1.1, 1, 1])
        date = c1.date_input("Date", value=pd.Timestamp("2018-06-15").date())
        hour = c2.selectbox(
            "Current hour", options=list(range(24)), index=17,
            format_func=lambda h: f"{h:02d}:00", key="hour",
        )
        holiday = c3.checkbox("Public holiday")

        st.markdown("**2 · Weather at that hour**")
        st.caption(
            "Weather is secondary to the daily traffic pattern, but snow and rain do pull "
            "volumes down."
        )
        w1, w2 = st.columns(2)
        weather = w1.selectbox("Conditions", weather_options)
        temp_c = w2.number_input("Temperature (°C)", value=20.0, step=0.5,
                                 min_value=-60.0, max_value=60.0)
        r1, r2, r3 = st.columns(3)
        rain = r1.number_input("Rain (mm)", value=0.0, step=0.5, min_value=0.0)
        snow = r2.number_input("Snow (mm)", value=0.0, step=0.5, min_value=0.0)
        clouds = r3.number_input("Cloud cover (%)", value=20.0, step=5.0,
                                 min_value=0.0, max_value=100.0)

        st.markdown("**3 · Last 25 hourly readings**")
        st.caption(
            "Vehicles per hour, oldest first, ending with the current hour. These 25 numbers "
            "become the lag and rolling features."
        )

        preset = st.radio(
            "Fill with", PRESETS, horizontal=True,
            help="Presets overwrite the grid. Edit any row afterwards; edits persist "
                 "until you change the hour or the preset.",
        )

        sig = (str(date), hour, preset)
        refresh(sig, preset, hour, date)

        frame = pd.DataFrame(
            {
                "hour": [f"{(hour - i) % 24:02d}:00" for i in range(24, -1, -1)],
                "vehicles/hour": st.session_state.readings,
            }
        )
        edited = st.data_editor(
            frame,
            hide_index=True,
            width="stretch",
            key="grid",
            disabled=["hour"],
            column_config={
                "hour": st.column_config.TextColumn("Hour"),
                "vehicles/hour": st.column_config.NumberColumn(
                    "Vehicles / hour", min_value=0, max_value=8000, step=1
                ),
            },
            height=380,
        )
        readings = [int(v) if pd.notna(v) else 0 for v in edited["vehicles/hour"]]
        st.session_state.readings = readings

    with right:
        payload = {
            "date": str(date),
            "hour": hour,
            "weather_main": weather,
            "temp_c": temp_c,
            "rain_1h": rain,
            "snow_1h": snow,
            "clouds_all": clouds,
            "is_holiday": bool(holiday),
            "history": readings,
        }

        try:
            result = ui.predict_from_payload(payload)
        except Exception as exc:
            st.error(f"Cannot predict from these values: {exc}")
            st.stop()

        icon = TIER_STYLE.get(result["tier"], "⚪")
        forecast_for = pd.Timestamp(result["forecast_for"])

        st.markdown("**Forecast**")
        st.caption(f"Next hour · {forecast_for:%a %d %b %Y} · {forecast_for:%H}:00")
        st.markdown(
            f"<div style='font-size:46px;font-weight:700;letter-spacing:-.03em;"
            f"line-height:1.05;margin:0 0 8px;font-variant-numeric:tabular-nums'>"
            f"{result['predicted']:,.0f}"
            f"<span style='font-size:13px;font-weight:500;color:#64748b;margin-left:6px'>"
            f"veh/h</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"{icon} **{result['tier']}**")
        st.markdown(f"<p class='meaning'>{result['summary']}</p>", unsafe_allow_html=True)
        st.markdown(tier_scale(meta, result["predicted"]), unsafe_allow_html=True)

        st.markdown(
            f"""<div class="stat-strip">
              <div><span class="k">Test R²</span><span class="v">{meta['model']['r2']:.3f}</span></div>
              <div><span class="k">Mean error</span><span class="v">±{meta['model']['mae']:,.0f}</span></div>
              <div><span class="k">Typical miss</span><span class="v">{meta['model'].get('mape', 6.1)}%</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.altair_chart(history_chart(readings, result["predicted"]), width="stretch")

        st.markdown(
            f"""<div class="kv"><span>Typical error</span>
              <span>±{result['typical_error']:,.0f} → likely {result['low']:,.0f}–{result['high']:,.0f}</span></div>
            <div class="kv"><span>Change vs current hour</span>
              <span>{result['current']:,.0f} → <b>{result['predicted']:,.0f}</b></span></div>""",
            unsafe_allow_html=True,
        )

        if result["actual"] is not None:
            st.markdown(
                f"""<div class="kv"><span>Recorded next hour</span>
                  <span><b>{result['actual']:,.0f}</b> veh/h · error {result['error']:,.0f}</span></div>""",
                unsafe_allow_html=True,
            )
            st.caption(
                "This hour exists in the 2018 test set, so the forecast can be checked "
                "against what the sensor actually recorded."
            )
        else:
            st.caption(
                "That hour is outside the recorded test period, so only the forecast is shown."
            )

        with st.expander("How this works", expanded=True):
            st.markdown(
                f"""
- The model is a **{meta['model']['name']}** trained on {meta['dataset']['records']}
  from one I-94 sensor; the fit stops at December 2017.
- Tested on {meta['dataset']['test_period']}, never shuffled: MAE
  {meta['model']['mae']} veh/h, RMSE {meta['model']['rmse']}, R² {meta['model']['r2']:.3f}.
- Your 25 readings and the weather build **{meta['features']} features**: 7 lags, 6 rolling
  statistics, calendar and weather. The current hour alone drives about 48% of the decision,
  which is why the grid is pre-filled with real data.
- Tiers: below {meta['thresholds']['free_max']:,} free-flow,
  {meta['thresholds']['free_max']:,}–{meta['thresholds']['moderate_max']:,} moderate,
  above {meta['thresholds']['moderate_max']:,} congested.
"""
            )

    st.divider()
    st.caption(
        f"{meta['dataset']['name']} · {meta['dataset']['source']} · "
        f"{meta['dataset']['licence']}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{meta['dataset']['station']}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;{meta['model']['name']} · test MAE "
        f"{meta['model']['mae']} veh/h · R² {meta['model']['r2']:.3f}",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
