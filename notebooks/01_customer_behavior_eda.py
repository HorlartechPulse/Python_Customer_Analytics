"""
Reproducible "notebook-style" script — Customer Behavior & Anomaly EDA
Run: python notebooks/01_customer_behavior_eda.py
Or convert to Jupyter: jupytext / copy cells manually.
"""
# %% [markdown]
# # Adult Income · Customer Behavior & Anomaly Analysis
# Case study pipeline: ingest → quality → pandas EDA → anomalies → charts

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ingest import load_raw
from quality import profile, clean, quality_report
from analysis import (
    high_income_by_category, occupation_high_earner_share,
    detect_anomalies, anomaly_summary, behavior_segments,
)
from visualize import *
import pandas as pd

# %%
raw = load_raw()
raw.head()

# %%
profile(raw.replace("?", pd.NA)).head(20)

# %%
df = clean(raw)
quality_report(raw, df)

# %%
high_income_by_category(df, "education").head(10)

# %%
occupation_high_earner_share(df)

# %%
plot_high_income_by_education(df)
plot_occupation_donut(occupation_high_earner_share(df))
plot_hours_distribution(df)

# %%
flagged = detect_anomalies(df)
print(anomaly_summary(flagged))
behavior_segments(df)
