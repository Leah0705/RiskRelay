"""
evaluate.py — standardized evaluation for all three models so results are
directly comparable. Reports both threshold-independent metrics (AUC-ROC,
AUC-PR — the latter matters much more than ROC-AUC under heavy class
imbalance) and threshold-dependent metrics (precision/recall/F1 at the
tuned operating point).
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
)


def evaluate_model(y_true, y_proba, threshold: float = 0.5, model_name: str = "model") -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)  # a.k.a. AUC-PR / average precision
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }

    print(f"\n=== {model_name} ===")
    print(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {cm}")

    return metrics


def compare_models(results: list, out_path: str = "reports/model_comparison.json"):
    df = pd.DataFrame(results)
    print("\n=== Model comparison ===")
    print(df[["model", "roc_auc", "pr_auc", "precision", "recall", "f1"]].to_string(index=False))

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison to {out_path}")
    return df
