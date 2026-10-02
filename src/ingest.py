"""Data ingestion for Adult Income / customer proxy dataset."""
from __future__ import annotations

import pandas as pd

from config import DATA_PATH

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Load CSV; handle UCI file with optional header and ' ?' missings."""
    df = pd.read_csv(path, skipinitialspace=True)
    # If no proper header
    if "age" not in df.columns:
        df = pd.read_csv(path, header=None, names=COLUMNS, skipinitialspace=True)
    # Strip column names
    df.columns = [c.strip().lower().replace("-", "_") for c in df.columns]
    # Normalize income label
    if "income" in df.columns:
        df["income"] = df["income"].astype(str).str.strip().str.replace(".", "", regex=False)
    return df


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df.head())
