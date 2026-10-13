"""
data_loader.py — loads event data and performs a TIME-BASED split.

Risk scoring must never use a random train/test split: risk patterns
drift over time, and a random split leaks future information into training
(e.g. an account's later "known risky" behavior leaking into earlier rows
via engineered rolling features). We always split chronologically.
"""

import sys
import numpy as np
import pandas as pd

sys.path.append(".")
from src.config import DATA
from data.generate_synthetic_data import attach_identity_fields


def load_raw(path: str = None) -> pd.DataFrame:
    path = path or DATA.raw_path
    df = pd.read_csv(path)

    # If this is a bare base-schema file (Time, V1-V28, Amount, Class only,
    # no identity columns), synthesize the identity fields so the full
    # feature set (velocity, counterparty risk, device consistency) can be built
    # on top of REAL labels/PCA features. See data/generate_synthetic_data.py
    # for the exact logic and its documented limitations.
    identity_cols = {"account_id", "counterparty_id", "device_id"}
    if not identity_cols.issubset(df.columns):
        n_accounts = max(1000, len(df) // 15)
        n_counterparties = max(200, len(df) // 100)
        n_devices = max(1000, len(df) // 12)
        df = attach_identity_fields(df, n_accounts, n_counterparties, n_devices)

    if "timestamp" not in df.columns:
        base = pd.Timestamp("2024-01-01")
        df["timestamp"] = base + pd.to_timedelta(df["Time"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, train_frac: float = None, val_frac: float = None):
    train_frac = train_frac or DATA.train_frac
    val_frac = val_frac or DATA.val_frac

    df = df.sort_values(DATA.time_col).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"Split sizes -> train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = part[DATA.target_col].mean()
        print(f"  {name} positive rate: {rate:.4%} ({part[DATA.target_col].sum():,} positives)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    df = load_raw()
    print(df.head())
    print(df.shape)
    time_based_split(df)
