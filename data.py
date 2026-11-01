"""Layer 1 - ingest and clean the raw dataset.

Contract of this module: `get_clean()` returns a DataFrame with
  * a DatetimeIndex, strictly hourly, sorted, unique
  * numeric columns: temp (K), rain_1h, snow_1h, clouds_all, traffic_volume
  * categorical columns: weather_main, weather_description
  * a binary is_holiday column
  * missing hours present as all-NaN rows (gaps are explicit, not skipped)

Every cleaning decision made here is reversible and documented in
REPORT_OUTLINE.md section 5.
"""
from __future__ import annotations

import gzip
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import config

NUMERIC_RAW = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
CATEGORICAL_RAW = ["holiday", "weather_main", "weather_description"]


# ------------------------------------------------------------------ download
def download(force: bool = False) -> Path:
    """Fetch the UCI archive once and leave a decompressed CSV behind."""
    if config.RAW_CSV.exists() and not force:
        return config.RAW_CSV

    zip_path = config.DATA_RAW / "metro_traffic.zip"
    print(f"[data] downloading {config.DATASET_NAME} ...")
    urllib.request.urlretrieve(config.DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.DATA_RAW)

    # The archive ships the CSV gzip-compressed.
    for gz in config.DATA_RAW.glob("*.csv.gz"):
        with gzip.open(gz, "rb") as src, open(config.RAW_CSV, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()

    if not config.RAW_CSV.exists():
        raise RuntimeError("download finished but the CSV was not found")

    return config.RAW_CSV


def load_raw() -> pd.DataFrame:
    return pd.read_csv(download())


# -------------------------------------------------------------------- clean
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw export into one tidy row per hour."""
    df = raw.copy()
    n_raw = len(df)
    df["date_time"] = pd.to_datetime(df["date_time"])

    # 1. exact duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_exact_dupes = before - len(df)

    # 2. repeated timestamps: the feed sometimes reports the same hour twice.
    #    Average the continuous readings, keep the first non-null category.
    agg: dict[str, object] = {c: "mean" for c in NUMERIC_RAW}
    for col in CATEGORICAL_RAW:
        agg[col] = lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan

    before = len(df)
    df = df.groupby("date_time", as_index=False).agg(agg)
    n_merged_timestamps = before - len(df)

    # 3. rebuild a complete hourly grid so absent hours become explicit gaps.
    grid = pd.date_range(df["date_time"].min(), df["date_time"].max(), freq="h")
    df = df.set_index("date_time").reindex(grid)
    df.index.name = "date_time"
    n_absent_hours = int(df["traffic_volume"].isna().sum())

    # 4. invalid readings
    n_bad_temp = int((df["temp"] <= config.TEMP_ABSOLUTE_ZERO).sum())
    df.loc[df["temp"] <= config.TEMP_ABSOLUTE_ZERO, "temp"] = np.nan

    n_bad_rain = int((df["rain_1h"] > config.RAIN_MAX_MM).sum())
    df["rain_1h"] = df["rain_1h"].clip(upper=config.RAIN_MAX_MM)

    # 5. a blank holiday column means "this is not a holiday"
    df["is_holiday"] = df["holiday"].notna().astype("int8")
    df = df.drop(columns=["holiday"])

    report = {
        "raw_rows": n_raw,
        "exact_duplicate_rows_removed": n_exact_dupes,
        "duplicate_timestamps_merged": n_merged_timestamps,
        "absent_hourly_slots": n_absent_hours,
        "invalid_temp_set_to_nan": n_bad_temp,
        "impossible_rain_capped": n_bad_rain,
        "rows_after_clean": len(df),
    }
    print("[data] cleaning report:")
    for k, v in report.items():
        print(f"         {k:34s} {v:,}")

    return df


# ----------------------------------------------------------------- validate
def validate(df: pd.DataFrame) -> None:
    """Fail loudly if the cleaned table breaks its contract."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")

    if not df.index.is_monotonic_increasing:
        raise ValueError("index is not sorted")

    if not df.index.is_unique:
        raise ValueError("index has duplicate timestamps")

    gaps = df.index.to_series().diff().dropna().unique()
    if not (len(gaps) == 1 and pd.Timedelta(gaps[0]) == pd.Timedelta(hours=1)):
        raise ValueError(f"index is not strictly hourly, found gaps: {gaps}")

    if (df["traffic_volume"].dropna() < 0).any():
        raise ValueError("negative traffic volume")

    expected = set(NUMERIC_RAW) | {"weather_main", "weather_description", "is_holiday"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns: {missing}")

    print(f"[data] validated: {len(df):,} hourly rows, "
          f"{df.index.min():%Y-%m-%d} -> {df.index.max():%Y-%m-%d}")


# --------------------------------------------------------------------- api
def get_clean(refresh: bool = False) -> pd.DataFrame:
    """Clean once, cache to CSV, reload thereafter."""
    if config.CLEAN_CSV.exists() and not refresh:
        df = pd.read_csv(config.CLEAN_CSV, index_col="date_time", parse_dates=True)
        validate(df)
        return df

    df = clean(load_raw())
    validate(df)
    df.to_csv(config.CLEAN_CSV)
    return df


if __name__ == "__main__":
    get_clean(refresh=True)
