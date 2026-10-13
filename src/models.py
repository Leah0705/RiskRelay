"""
models.py — model factory functions for the three-tier modeling approach:
  1. Logistic Regression: fast, interpretable baseline / sanity check
  2. Random Forest: stronger nonlinear baseline, good for feature-importance sanity check
  3. XGBoost: final production model — best accuracy/latency tradeoff, native
     imbalance handling via scale_pos_weight, and strong SHAP support
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import MODEL


def build_logistic_regression(class_weight=None):
    params = dict(MODEL.lr_params)
    return LogisticRegression(class_weight=class_weight, **params)


def build_random_forest(class_weight=None):
    params = dict(MODEL.rf_params)
    return RandomForestClassifier(class_weight=class_weight, **params)


def build_xgboost(scale_pos_weight: float = 1.0):
    params = dict(MODEL.xgb_params)
    return XGBClassifier(scale_pos_weight=scale_pos_weight, **params)
