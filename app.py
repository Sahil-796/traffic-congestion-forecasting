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
    "free-flow": ("🟢", "free-flow"),
    "moderate": ("🟠", "moderate"),
    "congested": ("🔴", "congested"),
}


@st.cache_resource(show_spinner="Loading model ...")
def boot() -> dict:
    """Load the model and the descriptive metadata exactly once per process."""
    meta = ui.meta()
    return {"meta": meta, "weather": ui.weather_options()}


def current_hour() -> int:
    return int(st.session_state.get("hour", 17))


def fill(kind: str, hour: int, date) -> list[int]:
    """25 volumes ending at `hour`, from a preset or the hourly means."""
    meta = boot()["meta"]
    labels = [(hour - i) % 24 for i in range(24, -1, -1)]

    if kind == "Quiet night":
        return [480 if 0 <= h <= 5 else 900 for h in labels]
    if kind == "Rush hour":
        return [5400 if 6 <= h <= 9 else (5600 if 16 <= h <= 18 else 3600) for h in labels]

    weekend = kind == "Typical weekend" or (
        kind == "Match this date" and date.weekday() >= 5
    )
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

    st.title("I-94 Traffic Forecast")
    st.caption(
        "Westbound Interstate 94, Minnesota DOT station 301, between Minneapolis and "
        "St Paul. Predicts the next hour's volume from the previous 24 hours, calendar "
        "position and weather."
    )

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.subheader("When")
        c1, c2, c3 = st.columns(3)
        date = c1.date_input("Date", value=pd.Timestamp("2018-06-15").date())
        hour = c2.selectbox(
            "Hour of day", options=list(range(24)), index=17,
            format_func=lambda h: f"{h:02d}:00", key="hour",
        )
        holiday = c3.checkbox("Public holiday")

        st.subheader("Weather at this hour")
        w1, w2 = st.columns(2)
        weather = w1.selectbox("Conditions", weather_options)
        temp_c = w2.number_input("Temperature (°C)", value=20.0, step=0.5,
                                 min_value=-60.0, max_value=60.0)
        r1, r2, r3 = st.columns(3)
        rain = r1.number_input("Rain (mm)", value=0.0, step=0.5, min_value=0.0)
        snow = r2.number_input("Snow (mm)", value=0.0, step=0.5, min_value=0.0)
        clouds = r3.number_input("Cloud cover (%)", value=20.0, step=5.0,
                                 min_value=0.0, max_value=100.0)

        st.subheader("Traffic readings")
        st.caption(
            "Vehicles per hour. The last row is the current hour, and the first is 24 "
            "hours earlier. These become the lag and rolling features the model uses."
        )

        preset = st.radio(
            "Fill with",
            ["Typical weekday", "Typical weekend", "Match this date",
             "Quiet night", "Rush hour"],
            horizontal=True,
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
            height=430,
        )
        readings = [int(v) if pd.notna(v) else 0 for v in edited["vehicles/hour"]]
        st.session_state.readings = readings

    with right:
        st.subheader("Forecast")

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

        icon, label = TIER_STYLE.get(result["tier"], ("⚪", result["tier"]))

        st.metric(
            "Next hour",
            f"{result['predicted']:,.0f} veh/h",
            delta=f"{result['delta']:+,.0f} vs now",
            delta_color="off",
        )
        st.markdown(f"### {icon} {label}")
        st.caption(
            f"Typical error ±{result['typical_error']:,.0f} → likely "
            f"{result['low']:,.0f}–{result['high']:,.0f} veh/h"
        )

        bars = pd.DataFrame(
            {
                "when": ["now", "next hour"],
                "veh/h": [result["current"], result["predicted"]],
            }
        )
        chart = (
            alt.Chart(bars)
            .mark_bar(size=110, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("when:N", sort=["now", "next hour"], title=None,
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("veh/h:Q", title=None, scale=alt.Scale(domainMin=0)),
                color=alt.Color(
                    "when:N", legend=None,
                    scale=alt.Scale(range=["#94a3b8", "#0f766e"]),
                ),
                tooltip=["when:N", alt.Tooltip("veh/h:Q", format=",")],
            )
            .properties(height=210)
        )
        st.altair_chart(chart, width="stretch")

        if result["actual"] is not None:
            st.info(
                f"**Recorded next hour:** {result['actual']:,.0f} veh/h "
                f"({result['actual_tier']})  \n"
                f"**Absolute error:** {result['error']:,.0f} veh/h"
            )
        else:
            st.info(
                "No recorded value for that hour in the dataset, so only the forecast "
                "is shown."
            )

        with st.expander("How this works"):
            st.markdown(
                f"""
The model is a **{meta['model']['name']}** trained on {meta['data_range'][0]} –
{meta['data_range'][1]}, evaluated on a held-out 2018 period.

| Metric | Value |
|---|---|
| MAE | {meta['model']['mae']} veh/h |
| RMSE | {meta['model']['rmse']} veh/h |
| R² | {meta['model']['r2']} |

The 25 readings become 7 lag features (0–24 h) and 6 rolling statistics. Your
current hour alone accounts for about 48% of the model's decisions, which is why
it is pre-filled with real data rather than left blank.

Congestion tiers: below {meta['thresholds']['free_max']:,} free-flow,
{meta['thresholds']['free_max']:,}–{meta['thresholds']['moderate_max']:,} moderate,
above {meta['thresholds']['moderate_max']:,} congested.
"""
            )

    st.divider()
    st.caption(
        f"I-94 westbound, MN DOT station 301 · model {meta['model']['name']} · "
        f"test MAE {meta['model']['mae']} veh/h · R² {meta['model']['r2']}"
    )


if __name__ == "__main__":
    main()
