"""Exploratory data analysis - figures for report section 4."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid", context="notebook")
DPI = 150


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.OUT_FIG / name, dpi=DPI)
    plt.close(fig)


def plot_missing_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """How complete is each year? (data-quality figure for section 4/5.)"""
    per_year = df.groupby(df.index.year)["traffic_volume"].agg(
        expected="size", present="count"
    )
    per_year["missing"] = per_year["expected"] - per_year["present"]
    per_year["missing_%"] = (per_year["missing"] / per_year["expected"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x=per_year.index.astype(str), y=per_year["missing_%"], ax=ax, color="#ef4444"
    )
    ax.set_title("Share of hourly slots with no reading, by year")
    ax.set_ylabel("% missing")
    ax.set_xlabel("year")
    _save(fig, "eda_missing_by_year.png")
    return per_year


def plot_target_distribution(df: pd.DataFrame):
    volume = df["traffic_volume"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(volume, bins=60, ax=axes[0], color="#2563eb")
    axes[0].set_title("Distribution of hourly traffic volume")
    axes[0].set_xlabel("vehicles / hour")

    sns.boxplot(x=volume, ax=axes[1], color="#93c5fd")
    axes[1].set_title("Box plot - note the bimodal shape")
    axes[1].set_xlabel("vehicles / hour")

    _save(fig, "eda_target_distribution.png")


def plot_volume_over_time(df: pd.DataFrame):
    monthly = df["traffic_volume"].resample("ME").mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly.index, monthly.values, color="#0f766e", lw=1.6)
    ax.set_title("Average monthly traffic volume (2012-2018)")
    ax.set_ylabel("mean vehicles / hour")
    _save(fig, "eda_volume_over_time.png")


def plot_hourly_profile(df: pd.DataFrame):
    d = df.dropna(subset=["traffic_volume"]).copy()
    d["hour"] = d.index.hour
    d["day_type"] = np.where(d.index.day_of_week >= 5, "weekend", "weekday")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=d, x="hour", y="traffic_volume", hue="day_type",
                 errorbar="ci", ax=ax)
    ax.set_title("Average traffic volume by hour of day")
    ax.set_ylabel("vehicles / hour")
    ax.set_xlabel("hour")
    _save(fig, "eda_hourly_profile.png")


def plot_weekday_profile(df: pd.DataFrame):
    d = df.dropna(subset=["traffic_volume"]).copy()
    d["day"] = d.index.day_name()

    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=d, x="day", y="traffic_volume", order=order,
                ax=ax, color="#a78bfa")
    ax.set_title("Traffic volume by day of week")
    ax.set_ylabel("vehicles / hour")
    ax.set_xlabel("")
    _save(fig, "eda_weekday_profile.png")


def plot_weather_effect(df: pd.DataFrame):
    d = df.dropna(subset=["traffic_volume"]).copy()
    order = d.groupby("weather_main")["traffic_volume"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=d, x="weather_main", y="traffic_volume", order=order,
                ax=ax, color="#f59e0b")
    ax.set_title("Traffic volume by weather condition")
    ax.set_ylabel("vehicles / hour")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "eda_weather_effect.png")


def plot_correlation(df: pd.DataFrame):
    cols = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation between numeric variables")
    _save(fig, "eda_correlation.png")
    return corr


def run_eda(df: pd.DataFrame) -> None:
    print("[eda] writing figures ...")
    per_year = plot_missing_by_year(df)
    plot_target_distribution(df)
    plot_volume_over_time(df)
    plot_hourly_profile(df)
    plot_weekday_profile(df)
    plot_weather_effect(df)
    corr = plot_correlation(df)

    per_year.to_csv(config.OUT_TAB / "eda_missing_by_year.csv")
    corr.round(3).to_csv(config.OUT_TAB / "eda_correlation.csv")

    summary = df["traffic_volume"].describe().round(2)
    summary.to_csv(config.OUT_TAB / "eda_target_summary.csv", header=["value"])
    print("[eda] done -> outputs/figures, outputs/tables")
