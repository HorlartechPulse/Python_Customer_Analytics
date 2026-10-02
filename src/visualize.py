"""Visualization helpers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from config import FIG_DIR


def _save(name: str):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    return path


def plot_high_income_by_education(df: pd.DataFrame):
    g = df.groupby("education")["high_income"].mean().sort_values(ascending=False).head(10) * 100
    plt.figure(figsize=(8, 4.5))
    colors = plt.cm.viridis_r([i / max(len(g), 1) for i in range(len(g))])
    g.plot(kind="bar", color=colors, edgecolor="white")
    plt.title("High-income rate (%) by education")
    plt.ylabel("% earning >50K")
    plt.xlabel("")
    plt.xticks(rotation=35, ha="right")
    return _save("high_income_by_education.png")


def plot_occupation_donut(occ: pd.DataFrame):
    plt.figure(figsize=(6, 6))
    vals = occ["high_rate_pct"]
    labels = [f"{o}\n{p:.1f}%" for o, p in zip(occ["occupation"], vals)]
    colors = ["#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B"]
    wedges, texts = plt.pie(
        vals, labels=None, startangle=90, colors=colors[: len(vals)],
        wedgeprops=dict(width=0.45, edgecolor="white"),
    )
    plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.title("Top occupations by % high earners")
    return _save("occupation_high_earners_donut.png")


def plot_hours_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    for label, color in [(0, "#4C78A8"), (1, "#F58518")]:
        subset = df.loc[df["high_income"] == label, "hours_per_week"]
        plt.hist(subset, bins=30, alpha=0.55, label=("<=50K" if label == 0 else ">50K"), color=color)
    plt.xlabel("Hours per week")
    plt.ylabel("Count")
    plt.title("Hours worked by income group")
    plt.legend()
    return _save("hours_by_income.png")


def plot_anomaly_counts(summary: dict):
    keys = ["extreme_hours", "capital_gain_outliers", "young_high_income"]
    vals = [summary[k] for k in keys]
    plt.figure(figsize=(7, 4))
    plt.bar(keys, vals, color=["#E45756", "#F58518", "#54A24B"])
    plt.title("Anomaly flag counts")
    plt.ylabel("Records")
    plt.xticks(rotation=15)
    return _save("anomaly_counts.png")


def plot_segment_sizes(seg: pd.DataFrame):
    plt.figure(figsize=(8, 4))
    plt.barh(seg["segment"], seg["n"], color="#4C78A8")
    plt.xlabel("People")
    plt.title("Behavior segments (counts)")
    plt.gca().invert_yaxis()
    return _save("behavior_segments.png")
