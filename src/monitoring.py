"""
monitoring.py — production monitoring for a deployed risk model. Two
distinct failure modes are tracked, since they require different responses:

  1. DATA DRIFT: the distribution of INPUT features shifts (e.g. average
     event amount rises, a new counterparty category
     appears). Detected via Population Stability Index (PSI) per feature.
     Response: investigate upstream data changes; may not need retraining.

  2. CONCEPT DRIFT / PERFORMANCE DECAY: the relationship between features
     and the label itself changes (attackers adopt a new pattern). Detected
     by tracking precision/recall against a lagged ground-truth label feed
     (review outcomes and confirmed reports arrive days/weeks after the
     event). Response: retrain.
"""

import numpy as np
import pandas as pd


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI < 0.1: no significant shift. 0.1-0.25: moderate shift, monitor.
    > 0.25: significant shift, investigate / consider retraining.
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_feature_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                                  feature_cols: list) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(
            reference_df[col].dropna().values, current_df[col].dropna().values
        )
        status = "stable" if psi < 0.1 else ("moderate_shift" if psi < 0.25 else "significant_shift")
        rows.append({"feature": col, "psi": round(psi, 4), "status": status})

    report = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return report


def compute_prediction_drift(reference_scores: np.ndarray, current_scores: np.ndarray) -> dict:
    """Tracks whether the model's OUTPUT distribution (risk probability
    scores) is shifting — a cheap early-warning signal that doesn't require
    waiting for delayed ground-truth labels.
    """
    psi = population_stability_index(reference_scores, current_scores)
    return {
        "score_psi": round(psi, 4),
        "reference_mean_score": float(np.mean(reference_scores)),
        "current_mean_score": float(np.mean(current_scores)),
        "reference_flagged_rate": float(np.mean(reference_scores >= 0.5)),
        "current_flagged_rate": float(np.mean(current_scores >= 0.5)),
        "status": "stable" if psi < 0.1 else ("moderate_shift" if psi < 0.25 else "significant_shift"),
    }


def compute_delayed_performance_report(y_true_delayed: np.ndarray, y_proba_delayed: np.ndarray,
                                        threshold: float) -> dict:
    """Once delayed ground-truth labels (confirmed review outcomes) become
    available for a past batch of events, recompute live performance
    metrics and compare against the metrics recorded at training time.
    Call this on a scheduled job (e.g. daily/weekly) as labels arrive.
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    y_pred = (y_proba_delayed >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true_delayed, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_delayed, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_delayed, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_delayed, y_proba_delayed)) if len(np.unique(y_true_delayed)) > 1 else None,
        "n_samples": int(len(y_true_delayed)),
    }
