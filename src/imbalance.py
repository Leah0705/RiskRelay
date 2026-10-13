"""
imbalance.py — strategies for handling extreme class imbalance (a ~0.1-0.2%
positive rate is typical). Only ONE of class_weight / SMOTE should generally be
used at a time; combining both tends to over-correct and hurts precision.
"""

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, f1_score

from src.config import IMBALANCE


def get_class_weight_ratio(y_train) -> float:
    """Returns scale_pos_weight for XGBoost / class_weight ratio for sklearn:
    (# negative) / (# positive). This tells the model to penalize missing a
    positive case proportionally to how rare positives are in the data.
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    ratio = n_neg / max(n_pos, 1)
    return ratio


def apply_smote(X_train, y_train, target_ratio: float = None, random_state: int = 42):
    """Oversamples the minority (positive) class using SMOTE up to a target
    ratio of minority:majority (default from config, e.g. 0.10 = 10%).

    IMPORTANT: SMOTE must be fit ONLY on the training split, after the
    train/val/test split, never before — otherwise synthetic points
    interpolated from val/test positive examples would leak into training.
    """
    from imblearn.over_sampling import SMOTE

    target_ratio = target_ratio or IMBALANCE.smote_target_ratio
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    desired_pos = int(n_neg * target_ratio)

    if desired_pos <= n_pos:
        print("SMOTE target ratio already met or exceeded by real data; skipping oversampling.")
        return X_train, y_train

    sampling_strategy = desired_pos / n_neg
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE: {n_pos:,} -> {(y_res == 1).sum():,} positive samples "
          f"(target ratio {target_ratio:.2%})")
    return X_res, y_res


def tune_threshold_for_f1(y_true, y_proba) -> dict:
    """Sweeps thresholds on the VALIDATION set (never test) to find the
    operating point that maximizes F1, and also reports the precision/recall
    at a couple of alternative business-driven operating points.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = np.argmax(f1_scores)

    result = {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1": float(f1_scores[best_idx]),
        "precision_at_best": float(precisions[best_idx]),
        "recall_at_best": float(recalls[best_idx]),
    }

    # Business-oriented alternative: recall-prioritized threshold
    # (e.g. minimum acceptable precision of 0.3, maximize recall subject to that)
    min_precision = 0.30
    eligible = precisions[:-1] >= min_precision
    if eligible.any():
        eligible_idx = np.where(eligible)[0]
        best_recall_idx = eligible_idx[np.argmax(recalls[:-1][eligible_idx])]
        result["high_recall_threshold"] = float(thresholds[best_recall_idx])
        result["high_recall_precision"] = float(precisions[best_recall_idx])
        result["high_recall_recall"] = float(recalls[best_recall_idx])

    return result
