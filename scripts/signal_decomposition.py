"""
signal_decomposition.py — why the headline PR-AUC is 0.11.

`data/generate_synthetic_data.py` builds labels in two passes:

  1. It plants `n_rows × positive_rate` positives whose PCA components are
     shifted, so they carry real separating structure.
  2. It then flips additional *normal* rows to positive, weighted by
     counterparty category risk and device novelty, so that `counterparty_risk_score` and
     `is_new_device` correlate with the label instead of being pure noise.

Pass 2 produces far more positives than pass 1, and those positives look exactly
like normal traffic in PCA space. The model is therefore graded largely on
labels that carry no signal it could possibly learn from V1–V28.

This script recovers the two populations by replaying the generator with the
same seed, then scores the trained model against each separately. It answers a
question the aggregate metrics cannot: is the pipeline weak, or is the label?

Reads the model saved by train.py; trains nothing. Writes
reports/signal_decomposition.json.

Run:
    python scripts/signal_decomposition.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

from generate_synthetic_data import (  # noqa: E402
    attach_identity_fields, generate_base_events,
)
from src.config import DATA, PATHS  # noqa: E402
from src.data_loader import load_raw, time_based_split  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    build_feature_pipeline, get_feature_columns,
)

OUT_PATH = "reports/signal_decomposition.json"

# The arguments the shipped data/events.csv was generated with. Replaying
# them reproduces that file byte-for-byte (see README, Reproduce it).
GEN = dict(n_rows=500_000, positive_rate=0.0017, n_accounts=20_000,
           n_counterparties=3_000, n_devices=25_000, seed=42)


def recover_populations() -> np.ndarray:
    """Replay the generator to recover which positives were planted.

    Returns a boolean array, True where the row was positive *before* the
    correlation-boost pass relabelled anything.
    """
    base = generate_base_events(
        GEN["n_rows"], GEN["positive_rate"], GEN["seed"])
    planted = base["Class"].values.astype(bool).copy()
    # attach_identity_fields does not touch Class or Time; it is replayed only
    # so the row order matches the shipped file exactly.
    base = attach_identity_fields(base, GEN["n_accounts"], GEN["n_counterparties"],
                                  GEN["n_devices"], GEN["seed"])
    shipped = pd.read_csv(DATA.raw_path, usecols=["Time", "Class"])
    if len(shipped) != len(base) or not np.allclose(
            shipped["Time"].values, base["Time"].values):
        raise SystemExit(
            "data/events.csv does not match the generator arguments in "
            "GEN — regenerate it with the command in the README, or update GEN.")
    return planted


def main():
    planted_all = recover_populations()

    df = load_raw(DATA.raw_path)
    df["_planted"] = planted_all
    train_df, val_df, test_df = time_based_split(df)
    # persist=False: never overwrite the encoders the serving path is using.
    train_df, val_df, test_df, _, _ = build_feature_pipeline(
        train_df, val_df, test_df, persist=False)

    feature_cols = get_feature_columns()
    y = test_df[DATA.target_col].values.astype(bool)
    planted = test_df["_planted"].values.astype(bool)
    flipped = y & ~planted

    model = joblib.load(PATHS.xgb_model_path)
    proba = model.predict_proba(test_df[feature_cols])[:, 1]

    # Score each positive population against the same negatives, so the only thing
    # that changes between the two numbers is which positives are present.
    negatives = ~y
    pr_all = average_precision_score(y, proba)
    pr_planted = average_precision_score(
        y[negatives | planted], proba[negatives | planted])
    pr_flipped = average_precision_score(
        y[negatives | flipped], proba[negatives | flipped])

    out = {
        "dataset": {
            "generator_args": GEN,
            "total_rows": int(len(df)),
            "total_positives": int(df[DATA.target_col].sum()),
            "planted_positives": int(planted_all.sum()),
            "flipped_positives": int(df[DATA.target_col].sum() - planted_all.sum()),
        },
        "test_split": {
            "rows": int(len(test_df)),
            "positives": int(y.sum()),
            "planted": int(planted.sum()),
            "flipped": int(flipped.sum()),
        },
        "pr_auc": {
            "all_positives": round(float(pr_all), 4),
            "planted_only": round(float(pr_planted), 4),
            "flipped_only": round(float(pr_flipped), 4),
        },
    }

    os.makedirs(PATHS.reports_dir, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    d, t, p = out["dataset"], out["test_split"], out["pr_auc"]
    print(f"\nDataset: {d['total_positives']:,} positives in {d['total_rows']:,} rows "
          f"({d['total_positives']/d['total_rows']:.2%})")
    print(f"  planted, PCA-shifted        : {d['planted_positives']:,} "
          f"({d['planted_positives']/d['total_positives']:.1%} of positives)")
    print(f"  flipped, correlation only   : {d['flipped_positives']:,} "
          f"({d['flipped_positives']/d['total_positives']:.1%} of positives)")
    print(f"\nTest split: {t['positives']:,} positives "
          f"= {t['planted']:,} planted + {t['flipped']:,} flipped")
    print(f"\nXGBoost PR-AUC, same negatives throughout:")
    print(f"  all positives               : {p['all_positives']:.4f}")
    print(f"  planted positives only      : {p['planted_only']:.4f}")
    print(f"  flipped positives only      : {p['flipped_only']:.4f}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
