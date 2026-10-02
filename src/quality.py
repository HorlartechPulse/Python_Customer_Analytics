"""Data quality checks."""
from __future__ import annotations

import pandas as pd
import numpy as np


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "nulls": int(s.isna().sum()),
            "null_pct": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "example": str(s.dropna().iloc[0]) if s.notna().any() else None,
        })
    return pd.DataFrame(rows)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standard cleaning for Adult dataset."""
    out = df.copy()
    # Treat '?' as missing
    out = out.replace("?", np.nan)
    out = out.replace(" ?", np.nan)
    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip()
        out.loc[out[col].isin(["nan", "None", "?", ""]), col] = np.nan
    # Drop rows with critical nulls
    critical = [c for c in ["age", "income", "hours_per_week"] if c in out.columns]
    out = out.dropna(subset=critical)
    # Types
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["hours_per_week"] = pd.to_numeric(out["hours_per_week"], errors="coerce")
    out["capital_gain"] = pd.to_numeric(out.get("capital_gain", 0), errors="coerce").fillna(0)
    out["capital_loss"] = pd.to_numeric(out.get("capital_loss", 0), errors="coerce").fillna(0)
    out = out.dropna(subset=["age", "hours_per_week"])
    out["high_income"] = (out["income"].astype(str).str.contains("50K") & out["income"].astype(str).str.contains(">")).astype(int)
    # Fix high_income detection
    out["high_income"] = out["income"].apply(lambda x: 1 if str(x).startswith(">") else 0)
    return out.reset_index(drop=True)


def quality_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> dict:
    return {
        "rows_raw": len(df_raw),
        "rows_clean": len(df_clean),
        "rows_dropped": len(df_raw) - len(df_clean),
        "duplicate_rows_raw": int(df_raw.duplicated().sum()),
        "high_income_rate": float(df_clean["high_income"].mean()) if "high_income" in df_clean else None,
        "null_pct_by_col_raw": df_raw.replace("?", np.nan).isna().mean().round(4).to_dict(),
    }
