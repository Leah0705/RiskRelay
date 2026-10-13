"""
explainability.py — SHAP-based explainability for the final XGBoost model.

Two use cases are covered:
  1. Global explainability: which features drive risk predictions overall
     (for model governance, validation and audit review).
  2. Local explainability: why THIS specific event was flagged (for
     reviewers triaging alerts).
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_shap_values(model, X_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def save_global_summary_plot(explainer, shap_values, X_sample: pd.DataFrame,
                              out_path: str = "reports/shap_summary.png"):
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved global SHAP summary plot to {out_path}")


def get_top_global_features(explainer, shap_values, X_sample: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return importance.head(top_n)


def explain_single_event(explainer, X_row: pd.DataFrame, top_n: int = 8) -> list:
    """Returns a human-readable list of the top contributing factors for a
    single event — this is what an API response / reviewer UI
    would surface for a flagged alert.
    """
    shap_vals = explainer.shap_values(X_row)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    contributions = pd.DataFrame({
        "feature": X_row.columns,
        "value": X_row.iloc[0].values,
        "shap_contribution": shap_vals,
    })
    contributions["abs_contribution"] = contributions["shap_contribution"].abs()
    contributions = contributions.sort_values("abs_contribution", ascending=False).head(top_n)

    explanations = []
    for _, row in contributions.iterrows():
        direction = "increased" if row["shap_contribution"] > 0 else "decreased"
        explanations.append({
            "feature": row["feature"],
            "value": float(row["value"]),
            "contribution": float(row["shap_contribution"]),
            "effect": f"{direction} risk",
        })
    return explanations
