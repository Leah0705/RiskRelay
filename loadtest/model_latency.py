"""
model_latency.py — per-model single-row inference latency.

loadtest.py measures the whole service: HTTP, feature store read, inference,
threshold, commit. That end-to-end number cannot answer the model-selection
question, because it only ever exercises the model that is actually serving.

This script isolates the inference call itself for all three trained models,
so "XGBoost was chosen partly on latency" is a measurement rather than an
assertion. Batch size is 1, because that is what inline, per-event scoring is.

Also reports each model at n_jobs=1 alongside its configured setting: at batch
size 1 the thread pool has almost nothing to divide, so the parallel setting
that helps during training does not necessarily help during serving.

Reads the models saved by train.py; trains nothing. Writes
reports/model_latency.json.

Run:
    python loadtest/model_latency.py
"""

import argparse
import json
import os
import platform
import sys
import time

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATA, PATHS
from src.data_loader import load_raw, time_based_split
from src.feature_engineering import build_feature_pipeline, get_feature_columns

OUT_PATH = "reports/model_latency.json"


def bench(model, X1, n_iter: int, n_warmup: int) -> dict:
    for _ in range(n_warmup):
        model.predict_proba(X1)

    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        model.predict_proba(X1)
        samples.append((time.perf_counter() - t0) * 1000.0)

    a = np.array(samples)
    return {
        "p50_ms": round(float(np.percentile(a, 50)), 3),
        "p95_ms": round(float(np.percentile(a, 95)), 3),
        "p99_ms": round(float(np.percentile(a, 99)), 3),
        "mean_ms": round(float(a.mean()), 3),
        "n_iter": n_iter,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--hardware", default="unknown")
    args = ap.parse_args()

    df = load_raw(DATA.raw_path)
    train_df, val_df, test_df = time_based_split(df)
    # persist=False: never overwrite the encoders the serving path is using.
    train_df, val_df, test_df, _, _ = build_feature_pipeline(
        train_df, val_df, test_df, persist=False)

    feature_cols = get_feature_columns()
    X1 = test_df[feature_cols].iloc[[0]]  # one event, the serving shape

    models = {
        "Logistic Regression": joblib.load(PATHS.lr_model_path),
        "Random Forest": joblib.load(PATHS.rf_model_path),
        "XGBoost": joblib.load(PATHS.xgb_model_path),
    }

    results = {}
    for name, model in models.items():
        configured = model.get_params().get("n_jobs")
        entry = {"configured_n_jobs": configured,
                 "as_configured": bench(model, X1, args.iterations, args.warmup)}

        single = joblib.load({"Logistic Regression": PATHS.lr_model_path,
                              "Random Forest": PATHS.rf_model_path,
                              "XGBoost": PATHS.xgb_model_path}[name])
        if "n_jobs" in single.get_params():
            single.set_params(n_jobs=1)
            entry["n_jobs_1"] = bench(single, X1, args.iterations, args.warmup)
        results[name] = entry

    out = {
        "environment": {
            "hardware": args.hardware,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "batch_size": 1,
        "models": results,
    }

    os.makedirs(PATHS.reports_dir, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSingle-row predict_proba, batch size 1, {os.cpu_count()} cores\n")
    header = f"{'model':<22}{'n_jobs':>8}{'p50 ms':>10}{'p99 ms':>10}"
    print(header)
    print("-" * len(header))
    for name, entry in results.items():
        print(f"{name:<22}{str(entry['configured_n_jobs']):>8}"
              f"{entry['as_configured']['p50_ms']:>10.3f}"
              f"{entry['as_configured']['p99_ms']:>10.3f}")
        if "n_jobs_1" in entry:
            print(f"{'  same, n_jobs=1':<22}{'1':>8}"
                  f"{entry['n_jobs_1']['p50_ms']:>10.3f}"
                  f"{entry['n_jobs_1']['p99_ms']:>10.3f}")
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
