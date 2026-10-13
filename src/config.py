"""
config.py — central configuration for the RiskRelay pipeline.
Keeping these in one place makes the pipeline reproducible and makes it
obvious what to tune when adapting to a new data source.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    raw_path: str = "data/events.csv"
    train_frac: float = 0.7
    val_frac: float = 0.15
    test_frac: float = 0.15  # inferred as remainder; kept for clarity
    time_col: str = "timestamp"
    target_col: str = "Class"
    random_state: int = 42


@dataclass
class FeatureConfig:
    velocity_windows_minutes: List[int] = field(default_factory=lambda: [10, 60, 1440])  # 10m, 1h, 24h
    pca_cols: List[str] = field(default_factory=lambda: [f"V{i}" for i in range(1, 29)])
    amount_col: str = "Amount"
    account_col: str = "account_id"
    counterparty_col: str = "counterparty_id"
    counterparty_category_col: str = "counterparty_category"
    device_col: str = "device_id"


@dataclass
class ImbalanceConfig:
    strategy: str = "class_weight"  # one of: "class_weight", "smote", "none"
    smote_target_ratio: float = 0.10  # SMOTE brings minority class up to 10% of majority


@dataclass
class ModelConfig:
    xgb_params: dict = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
    })
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": 42,
    })

    # Training and serving want opposite thread settings, so they are separate.
    #
    # Training fits 350k rows at once and should use every core (n_jobs=-1
    # above). Serving scores ONE row per request, where a thread pool has almost
    # nothing to divide: on the 2-vCPU box behind reports/model_latency.json the
    # two settings stay within run-to-run noise (1.4-1.6 ms single-row p50).
    #
    # The scale-out argument is the reason for the setting. Horizontal scaling
    # here means N uvicorn workers behind a load balancer; if each worker's
    # inference claims all cores, N workers oversubscribe the box and contend
    # rather than multiply. Pinning serving to one thread per worker is what
    # makes "one worker per core" actually scale.
    #
    # Applied in src/api.py at model load. Set to 0 or None to leave whatever
    # the model was trained with untouched.
    serving_n_jobs: int = 1
    lr_params: dict = field(default_factory=lambda: {
        "max_iter": 1000,
        "random_state": 42,
    })


@dataclass
class PathsConfig:
    model_dir: str = "models"
    reports_dir: str = "reports"
    feature_list_path: str = "models/feature_list.joblib"
    counterparty_risk_map_path: str = "models/counterparty_risk_map.joblib"
    device_map_path: str = "models/device_map.joblib"
    xgb_model_path: str = "models/xgb_risk_model.joblib"
    rf_model_path: str = "models/rf_risk_model.joblib"
    lr_model_path: str = "models/lr_risk_model.joblib"
    threshold_path: str = "models/decision_threshold.joblib"


@dataclass
class RegistryConfig:
    """MLflow-backed model registry. Everything is local: a file-based
    tracking/registry backend under ./mlruns, no server and no cloud. The
    registry — not the joblib files in models/ — is the source of truth for
    which model is serving. joblibs are kept only as a thin compatibility shim.

    Stages follow MLflow's two canonical registry stages:
      - "Staging":    a freshly trained (or challenger) version, not yet serving.
      - "Production": the single version the API serves right now (the champion).
    Promotion moves a version to Production and archives the previous one;
    rollback is just promoting an older version back to Production.
    """
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "riskrelay"
    registered_model_name: str = "risk_xgb"
    staging_stage: str = "Staging"
    production_stage: str = "Production"


@dataclass
class MonitoringConfig:
    """Cadence and thresholds for the running monitoring service. PSI bands
    match src/monitoring.py exactly (0.1 moderate, 0.25 significant) so the
    scheduled service and the ad-hoc functions never disagree on what counts
    as a shift.
    """
    interval_seconds: int = 3600          # one drift cycle per hour by default
    batch_size: int = 5000                # recent events pulled per cycle
    history_dir: str = "reports/monitoring"
    psi_moderate: float = 0.10
    psi_significant: float = 0.25
    # How many features must individually cross the significant band before the
    # (feature) drift alert fires. One noisy feature is not an incident.
    min_significant_features: int = 3
    # Realized-performance floors for the delayed-label job. Absolute floors are
    # deliberately low because the synthetic data's signal is weak — the point is
    # the mechanism, not the number. See README "Model lifecycle".
    realized_f1_floor: float = 0.05
    realized_recall_floor: float = 0.30
    # Optional alert webhook. NEVER hardcode an endpoint here — it is read from
    # the RISK_ALERT_WEBHOOK environment variable at runtime and left blank
    # (log-only) if unset.
    webhook_env_var: str = "RISK_ALERT_WEBHOOK"


@dataclass
class RetrainingConfig:
    """Policy for the automated retraining trigger and the champion/challenger
    promotion gate. All thresholds live here so the retrain decision is
    auditable and tunable without touching code.
    """
    # ---- primary trigger: realized performance decay (concept drift) ----
    # Retrain if realized F1 falls below the absolute floor, OR drops by more
    # than this fraction relative to the F1 recorded at the champion's training.
    decay_f1_abs_floor: float = 0.05
    decay_f1_relative_drop: float = 0.25
    # ---- secondary/early trigger: sustained input/score drift ----
    # Significant drift must persist for this many consecutive cycles before it
    # alone triggers a retrain (a single spike is investigated, not acted on).
    drift_sustained_cycles: int = 2
    drift_min_significant_features: int = 3
    # ---- the promotion gate ----
    # Primary comparison metric and the margin the challenger must beat the
    # champion by on the common held-out test split. PR-AUC (not ROC-AUC)
    # because it is the metric that matters under extreme class imbalance.
    primary_metric: str = "pr_auc"
    promotion_min_pr_auc_gain: float = 0.005
    # Fraction (newest slice) of the available data used to train a challenger,
    # so the challenger learns the *recent* regime that triggered the retrain.
    challenger_window_frac: float = 0.5
    decisions_log_path: str = "reports/monitoring/retraining_decisions.jsonl"


DATA = DataConfig()
FEATURES = FeatureConfig()
IMBALANCE = ImbalanceConfig()
MODEL = ModelConfig()
PATHS = PathsConfig()
REGISTRY = RegistryConfig()
MONITORING = MonitoringConfig()
RETRAIN = RetrainingConfig()
