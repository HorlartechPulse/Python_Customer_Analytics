"""pandas analysis: behavior segments and anomalies."""
from __future__ import annotations

import pandas as pd
import numpy as np


def high_income_by_category(df: pd.DataFrame, col: str, min_n: int = 50) -> pd.DataFrame:
    g = df.groupby(col, dropna=False).agg(
        n=("high_income", "size"),
        high_income_rate=("high_income", "mean"),
        avg_hours=("hours_per_week", "mean"),
        avg_age=("age", "mean"),
    ).reset_index()
    g = g[g["n"] >= min_n].sort_values("high_income_rate", ascending=False)
    g["high_income_rate"] = (g["high_income_rate"] * 100).round(2)
    g["avg_hours"] = g["avg_hours"].round(1)
    g["avg_age"] = g["avg_age"].round(1)
    return g


def occupation_high_earner_share(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    sub = df.groupby("occupation").agg(
        n=("high_income", "size"),
        high_rate=("high_income", "mean"),
    )
    sub = sub[sub["n"] >= 100]
    sub = sub.sort_values("high_rate", ascending=False).head(top_n)
    sub["high_rate_pct"] = (sub["high_rate"] * 100).round(1)
    return sub.reset_index()


def correlation_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df[["age", "education_num", "capital_gain", "capital_loss", "hours_per_week", "high_income"]].copy()
    return num.corr().round(3)


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag statistical anomalies: extreme hours, capital gain outliers, age extremes."""
    out = df.copy()
    # Z-score on hours
    mu, sigma = out["hours_per_week"].mean(), out["hours_per_week"].std(ddof=0)
    out["hours_z"] = (out["hours_per_week"] - mu) / (sigma if sigma else 1)
    # IQR on capital_gain
    q1, q3 = out["capital_gain"].quantile(0.25), out["capital_gain"].quantile(0.75)
    iqr = q3 - q1
    out["anomaly_extreme_hours"] = (out["hours_z"].abs() > 3).astype(int)
    out["anomaly_capital_gain"] = (out["capital_gain"] > q3 + 3 * iqr).astype(int)
    out["anomaly_young_high_income"] = ((out["age"] < 25) & (out["high_income"] == 1)).astype(int)
    out["anomaly_score"] = (
        out["anomaly_extreme_hours"] + out["anomaly_capital_gain"] + out["anomaly_young_high_income"]
    )
    return out


def anomaly_summary(df_flagged: pd.DataFrame) -> dict:
    return {
        "extreme_hours": int(df_flagged["anomaly_extreme_hours"].sum()),
        "capital_gain_outliers": int(df_flagged["anomaly_capital_gain"].sum()),
        "young_high_income": int(df_flagged["anomaly_young_high_income"].sum()),
        "any_anomaly": int((df_flagged["anomaly_score"] > 0).sum()),
        "anomaly_rate_pct": round(100 * (df_flagged["anomaly_score"] > 0).mean(), 2),
    }


def behavior_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Simple rule-based customer/personas behavior segments."""
    def seg(row):
        if row["high_income"] == 1 and row["hours_per_week"] >= 45:
            return "High earner · intensive hours"
        if row["high_income"] == 1:
            return "High earner · standard hours"
        if row["hours_per_week"] >= 50:
            return "Long hours · not high income"
        if row["age"] < 30:
            return "Early career"
        return "Core workforce"
    s = df.copy()
    s["segment"] = s.apply(seg, axis=1)
    return s.groupby("segment").agg(
        n=("age", "size"),
        high_income_rate=("high_income", "mean"),
        avg_age=("age", "mean"),
        avg_hours=("hours_per_week", "mean"),
    ).reset_index().sort_values("n", ascending=False)
