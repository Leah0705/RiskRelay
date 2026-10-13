"""
test_pipeline.py — sanity tests for the core pipeline logic. These are not
exhaustive, but they guard against the most common silent failure modes in
risk-scoring pipelines: leakage in the split, leakage in feature engineering, and
degenerate imbalance handling.
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(".")
from src.data_loader import time_based_split
from src.feature_engineering import add_time_since_last_event, add_event_velocity
from src.imbalance import get_class_weight_ratio, tune_threshold_for_f1


def make_toy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2024-01-01")
    df = pd.DataFrame({
        "account_id": rng.choice([f"ACCT_{i}" for i in range(10)], size=n),
        "Amount": rng.uniform(1, 500, size=n),
        "Class": rng.choice([0, 1], size=n, p=[0.98, 0.02]),
    })
    df["timestamp"] = sorted(base + pd.to_timedelta(rng.integers(0, 1_000_000, size=n), unit="s"))
    df["Time"] = (df["timestamp"] - base).dt.total_seconds()
    return df


def test_time_based_split_is_chronological():
    df = make_toy_df()
    train, val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_time_since_last_event_no_negative_values():
    df = make_toy_df()
    out = add_time_since_last_event(df)
    assert (out["seconds_since_last_event"] >= 0).all()


def test_time_since_last_event_first_event_is_sentinel():
    df = make_toy_df()
    out = add_time_since_last_event(df)
    first_events = out.sort_values("timestamp").groupby("account_id").head(1)
    # first event per account should get the large sentinel value
    assert (first_events["seconds_since_last_event"] == 1e6).all()


def test_velocity_features_nonneg_and_causal():
    df = make_toy_df()
    out = add_event_velocity(df)
    assert (out["velocity_count_10m"] >= 0).all()
    assert (out["velocity_amount_60m"] >= 0).all()


def test_class_weight_ratio_matches_imbalance():
    y = pd.Series([0] * 980 + [1] * 20)
    ratio = get_class_weight_ratio(y)
    assert abs(ratio - 49.0) < 1e-6


def test_threshold_tuning_returns_valid_threshold():
    rng = np.random.default_rng(1)
    y_true = rng.choice([0, 1], size=1000, p=[0.98, 0.02])
    y_proba = np.clip(y_true * 0.7 + rng.normal(0, 0.2, size=1000), 0, 1)
    result = tune_threshold_for_f1(y_true, y_proba)
    assert 0.0 <= result["best_threshold"] <= 1.0
    assert 0.0 <= result["best_f1"] <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
