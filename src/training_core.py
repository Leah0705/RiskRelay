"""
training_core.py — the single, reusable definition of "train the production
XGBoost model on a dataset."

Both the full training entry point (train.py) and the automated challenger
retrain (src/retraining.py) call `train_xgb_pipeline` so a challenger is trained
by *exactly* the same leakage-safe, causal procedure as the champion — same
time-based split, same feature engineering, same imbalance handling, same
threshold-tuning-on-validation. If a challenger could be trained differently, a
gate comparison between champion and challenger would be meaningless.

This module deliberately does NOT touch the registry — it just returns a bundle.
Logging/registration is the caller's job (train.py, retraining.py), keeping the
training logic pure and importable/testable on its own.
"""

import numpy as np

from src.config import DATA, MODEL
from src.data_loader import time_based_split
from src.feature_engineering import build_feature_pipeline, get_feature_columns
from src.imbalance import get_class_weight_ratio, apply_smote, tune_threshold_for_f1
from src.models import build_xgboost
from src.evaluate import evaluate_model


def train_xgb_pipeline(df, imbalance_strategy: str = "class_weight",
                       seed: int = None, persist_artifacts: bool = True) -> dict:
    """Train the final XGBoost model on `df` end-to-end and return a bundle.

    Steps (identical to the production pipeline):
      1. chronological train/val/test split,
      2. leakage-safe feature engineering (encoders fit on train only),
      3. imbalance handling (class_weight / smote / none),
      4. fit XGBoost,
      5. tune the decision threshold on validation (F1-optimal),
      6. evaluate on the held-out, chronologically-last test split.

    Returns a dict with the fitted model, the serving sidecars (feature list,
    threshold, encoders), the test metrics, the logged params, and the
    featurized held-out test split (X_test/y_test) so a caller can score a
    *different* model on the same common test set for a champion/challenger
    comparison.

    `persist_artifacts=False` prevents the feature pipeline from overwriting the
    champion's on-disk joblibs (used when training a challenger).
    """
    seed = DATA.random_state if seed is None else seed

    train_df, val_df, test_df = time_based_split(df)
    train_df, val_df, test_df, risk_maps, device_map = build_feature_pipeline(
        train_df, val_df, test_df, persist=persist_artifacts
    )
    feature_cols = get_feature_columns()

    X_train, y_train = train_df[feature_cols], train_df[DATA.target_col]
    X_val, y_val = val_df[feature_cols], val_df[DATA.target_col]
    X_test, y_test = test_df[feature_cols], test_df[DATA.target_col]

    # ---- imbalance handling (mirrors train.py exactly) ----
    scale_pos_weight = get_class_weight_ratio(y_train)
    X_train_model, y_train_model = X_train, y_train
    xgb_scale_pos_weight = 1.0
    if imbalance_strategy == "class_weight":
        xgb_scale_pos_weight = scale_pos_weight
    elif imbalance_strategy == "smote":
        X_train_model, y_train_model = apply_smote(X_train, y_train, random_state=seed)
        xgb_scale_pos_weight = 1.0
    # "none": leave as-is

    # ---- fit + threshold tuning on validation ----
    xgb_params = dict(MODEL.xgb_params)
    xgb_params["random_state"] = seed
    xgb = build_xgboost(scale_pos_weight=xgb_scale_pos_weight)
    xgb.set_params(random_state=seed)
    xgb.fit(X_train_model, y_train_model, eval_set=[(X_val, y_val)], verbose=False)

    xgb_val_proba = xgb.predict_proba(X_val)[:, 1]
    threshold_info = tune_threshold_for_f1(y_val, xgb_val_proba)
    best_threshold = threshold_info["best_threshold"]

    # ---- held-out test metrics at the tuned operating point ----
    xgb_test_proba = xgb.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(
        y_test, xgb_test_proba, threshold=best_threshold,
        model_name="XGBoost (tuned threshold)",
    )
    metrics = {k: v for k, v in metrics.items()
               if k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}

    params = {
        "imbalance_strategy": imbalance_strategy,
        "seed": seed,
        "scale_pos_weight": round(float(xgb_scale_pos_weight), 4),
        "feature_cols": feature_cols,
        "xgb": {k: xgb_params[k] for k in
                ("n_estimators", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "min_child_weight", "gamma",
                 "reg_alpha", "reg_lambda")},
    }

    return {
        "model": xgb,
        "feature_cols": feature_cols,
        "threshold_info": threshold_info,
        "risk_maps": risk_maps,
        "device_map": device_map,
        "metrics": metrics,
        "params": params,
        "X_test": X_test,
        "y_test": y_test,
        "test_df": test_df,
    }


def score_on_common_test(model, X_test, y_test, threshold: float, model_name: str = "model") -> dict:
    """Score an already-fitted model on a common held-out test split and return
    the same metric dict shape as train_xgb_pipeline. Used to evaluate a
    champion on the challenger's test window (and vice versa) so both are judged
    on identical rows.
    """
    proba = model.predict_proba(X_test[[c for c in X_test.columns]])[:, 1] \
        if hasattr(X_test, "columns") else model.predict_proba(X_test)[:, 1]
    m = evaluate_model(np.asarray(y_test), proba, threshold=threshold, model_name=model_name)
    return {k: v for k, v in m.items()
            if k in ("roc_auc", "pr_auc", "precision", "recall", "f1")}
