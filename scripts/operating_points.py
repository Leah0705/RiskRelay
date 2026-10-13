"""
operating_points.py — turn the held-out test set into a review-queue table.

The model comparison in reports/model_comparison.json answers "how good is the
model". It does not answer the question this project is actually built around:
*how many alerts a day does each operating point put in front of a reviewer,
and how many of them are real?*

That number is already implicit in the confusion matrices — it is just
TP + FP, divided by the length of the test window — but it is the number the
whole capacity argument rests on, so it gets computed explicitly here rather
than left as an exercise.

Reads the models saved by train.py; trains nothing. Writes
reports/operating_points.json.

Run:
    python scripts/operating_points.py
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA, PATHS
from src.data_loader import load_raw, time_based_split
from src.feature_engineering import build_feature_pipeline, get_feature_columns

OUT_PATH = "reports/operating_points.json"


def test_window_days(test_df: pd.DataFrame) -> float:
    """Length of the test window in days.

    Uses min/max rather than the first and last row: the feature pipeline
    groups by account, so row order is no longer chronological by the time
    the frame comes back.
    """
    ts = pd.to_datetime(test_df[DATA.time_col])
    return (ts.max() - ts.min()).total_seconds() / 86400.0


def score_point(y_true, y_proba, threshold: float, model_name: str,
                span_days: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    alerts = int(tp + fp)

    return {
        "model": model_name,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "alerts": alerts,
        "alerts_per_day": round(alerts / span_days, 1),
        "false_alarms_per_catch": round(fp / tp, 1) if tp else None,
        "positives_caught_per_day": round(int(tp) / span_days, 1),
    }


def main():
    df = load_raw(DATA.raw_path)
    train_df, val_df, test_df = time_based_split(df)

    # persist=False: this script must never overwrite the encoders that the
    # serving path is currently using.
    train_df, val_df, test_df, _, _ = build_feature_pipeline(
        train_df, val_df, test_df, persist=False)

    feature_cols = get_feature_columns()
    X_test = test_df[feature_cols]
    y_test = test_df[DATA.target_col]
    span_days = test_window_days(test_df)

    lr = joblib.load(PATHS.lr_model_path)
    rf = joblib.load(PATHS.rf_model_path)
    xgb = joblib.load(PATHS.xgb_model_path)
    thresholds = joblib.load(PATHS.threshold_path)

    lr_proba = lr.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    points = [
        score_point(y_test, lr_proba, 0.5, "Logistic Regression", span_days),
        score_point(y_test, rf_proba, 0.5, "Random Forest", span_days),
        score_point(y_test, xgb_proba, 0.5, "XGBoost (default 0.5)", span_days),
        score_point(y_test, xgb_proba, thresholds["best_threshold"],
                    "XGBoost (F1-optimal)", span_days),
        score_point(y_test, xgb_proba, thresholds["high_recall_threshold"],
                    "XGBoost (precision-constrained)", span_days),
    ]

    out = {
        "test_window_days": round(span_days, 4),
        "test_events": int(len(test_df)),
        "test_positives": int(y_test.sum()),
        "test_positive_rate": float(y_test.mean()),
        "events_per_day": round(len(test_df) / span_days),
        "operating_points": points,
    }

    os.makedirs(PATHS.reports_dir, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nTest window: {out['test_window_days']} days | "
          f"{out['test_events']:,} events "
          f"({out['events_per_day']:,}/day) | "
          f"{out['test_positives']:,} positives ({out['test_positive_rate']:.4%})\n")
    header = (f"{'operating point':<34}{'thr':>7}{'prec':>8}{'recall':>8}"
              f"{'alerts/day':>12}{'FP per catch':>14}")
    print(header)
    print("-" * len(header))
    for p in points:
        print(f"{p['model']:<34}{p['threshold']:>7.3f}{p['precision']:>8.4f}"
              f"{p['recall']:>8.4f}{p['alerts_per_day']:>12,.1f}"
              f"{p['false_alarms_per_catch']:>14}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
