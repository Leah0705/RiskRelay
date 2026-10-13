"""
monitoring_service.py — turns the monitoring *functions* in src/monitoring.py
into a running *service*: something that executes on a cadence, persists a
timestamped history, and alerts when a band is crossed.

src/monitoring.py answers "how much has X drifted?" for a single batch. That is
necessary but not sufficient in production: nobody is standing there calling it.
This module schedules those checks, writes their results to disk so you can plot
decay over time, and raises alerts — keeping the two failure modes strictly
separate, because they demand different responses:

  * DRIFT alert (inputs / scores shifted): available immediately, no labels
    needed. Response: investigate upstream data; maybe an early retrain.
  * DECAY alert (realized precision/recall/F1 dropped once delayed labels
    arrived): the model actually got worse. Response: retrain, now.

Persistence is JSON Lines under reports/monitoring/ (one append per cycle) —
local, greppable, trivially plottable, no database to stand up. The decision to
ACT on these signals lives in src/retraining.py; this module only observes,
records, and alerts.

Scheduling uses APScheduler when available and falls back to a plain interval
loop, but all real logic lives in importable functions (run_monitoring_cycle /
run_delayed_performance_cycle) — nothing critical is buried in __main__ — so the
service is testable without a running scheduler.
"""

import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.config import MONITORING
from src.monitoring import (
    compute_feature_drift_report,
    compute_prediction_drift,
    compute_delayed_performance_report,
)

logger = logging.getLogger("riskrelay.monitoring")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

DRIFT_HISTORY = "drift_history.jsonl"
PERFORMANCE_HISTORY = "performance_history.jsonl"


# --------------------------------------------------------------------------- #
# Persistence (JSON Lines under reports/monitoring/)
# --------------------------------------------------------------------------- #
def _now_iso(ts=None) -> str:
    ts = ts or datetime.now(timezone.utc)
    return ts.isoformat()


def _history_path(filename: str, config=MONITORING) -> str:
    os.makedirs(config.history_dir, exist_ok=True)
    return os.path.join(config.history_dir, filename)


def append_history(filename: str, record: dict, config=MONITORING) -> str:
    """Append one JSON record (one line) to a history file. Returns the path."""
    path = _history_path(filename, config)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path


def load_history(filename: str, config=MONITORING) -> list:
    """Read a history file back as a list of records (empty if none yet)."""
    path = _history_path(filename, config)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Alerting — drift and decay are kept as distinct alert types on purpose
# --------------------------------------------------------------------------- #
def _post_webhook(url: str, payload: dict) -> None:
    """Best-effort POST to an operator-configured webhook. Failures are logged,
    never raised — a flaky alerting sink must not take down the monitor.
    """
    try:
        import httpx
        httpx.post(url, json=payload, timeout=5.0)
    except Exception as e:  # noqa: BLE001 — alerting must never crash monitoring
        logger.warning("alert webhook POST failed: %s", e)


def emit_alert(alert_type: str, severity: str, message: str, payload: dict,
               config=MONITORING) -> dict:
    """Emit a structured alert: always a structured log line; additionally a
    webhook POST if RISK_ALERT_WEBHOOK (config.webhook_env_var) is set. The
    endpoint is NEVER hardcoded — it comes from the environment or nothing.
    """
    alert = {
        "alert_type": alert_type,   # "drift" | "decay"
        "severity": severity,       # "warning" | "critical"
        "message": message,
        "timestamp": _now_iso(),
        **payload,
    }
    logger.warning("ALERT %s/%s: %s | %s", alert_type, severity, message,
                   json.dumps({k: v for k, v in payload.items() if k != "feature_drift"}))
    url = os.environ.get(config.webhook_env_var, "").strip()
    if url:
        _post_webhook(url, alert)
    return alert


# --------------------------------------------------------------------------- #
# Cycle 1 — drift (inputs + prediction scores). No labels required.
# --------------------------------------------------------------------------- #
def run_monitoring_cycle(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                         feature_cols: list,
                         reference_scores=None, current_scores=None,
                         config=MONITORING, timestamp=None, emit: bool = True) -> dict:
    """Run one drift-monitoring cycle and persist it.

    Computes per-feature PSI (input drift) against the training-distribution
    reference, and — if scores are supplied — prediction-score PSI (a cheap
    early-warning signal that needs no ground truth). Persists a timestamped
    record to drift_history.jsonl and raises a DRIFT alert when the significant
    band is crossed (enough features shift, or the score distribution shifts).

    Returns the persisted record. Pure aside from the append + optional alert,
    so it is directly testable.
    """
    feature_report = compute_feature_drift_report(reference_df, current_df, feature_cols)
    significant = feature_report[feature_report["status"] == "significant_shift"]
    n_significant = int(len(significant))

    score_drift = None
    if reference_scores is not None and current_scores is not None:
        score_drift = compute_prediction_drift(
            np.asarray(reference_scores), np.asarray(current_scores)
        )

    score_significant = bool(score_drift and score_drift["status"] == "significant_shift")
    drift_alert = (n_significant >= config.min_significant_features) or score_significant

    record = {
        "timestamp": _now_iso(timestamp),
        "cycle": "drift",
        "n_features_checked": int(len(feature_report)),
        "n_significant_features": n_significant,
        "significant_features": significant["feature"].tolist(),
        "top_feature_psi": feature_report.head(5).to_dict(orient="records"),
        "score_drift": score_drift,
        "drift_alert": drift_alert,
    }
    append_history(DRIFT_HISTORY, record, config)

    if emit and drift_alert:
        reasons = []
        if n_significant >= config.min_significant_features:
            reasons.append(f"{n_significant} features in significant PSI band "
                           f"(>= {config.min_significant_features})")
        if score_significant:
            reasons.append(f"prediction-score PSI significant "
                           f"({score_drift['score_psi']})")
        emit_alert(
            "drift", "warning",
            "Input/score drift crossed the significant band: " + "; ".join(reasons),
            {"n_significant_features": n_significant,
             "significant_features": record["significant_features"],
             "score_drift": score_drift},
            config,
        )
    return record


# --------------------------------------------------------------------------- #
# Cycle 2 — realized performance, once delayed labels arrive. Detects DECAY.
# --------------------------------------------------------------------------- #
def run_delayed_performance_cycle(y_true_delayed, y_proba_delayed, threshold: float,
                                  batch_id: str = None, config=MONITORING,
                                  timestamp=None, emit: bool = True) -> dict:
    """Run the delayed-ground-truth performance job for one past batch and
    persist realized precision/recall/F1 over time.

    Raises a DECAY alert (distinct from a drift alert) when realized F1 or recall
    falls below its configured floor — the signal that the model has genuinely
    degraded and should be retrained.
    """
    report = compute_delayed_performance_report(
        np.asarray(y_true_delayed), np.asarray(y_proba_delayed), threshold
    )
    below_f1 = report["f1"] < config.realized_f1_floor
    below_recall = report["recall"] < config.realized_recall_floor
    decay_alert = below_f1 or below_recall

    record = {
        "timestamp": _now_iso(timestamp),
        "cycle": "delayed_performance",
        "batch_id": batch_id,
        "threshold": float(threshold),
        **report,
        "f1_floor": config.realized_f1_floor,
        "recall_floor": config.realized_recall_floor,
        "decay_alert": decay_alert,
    }
    append_history(PERFORMANCE_HISTORY, record, config)

    if emit and decay_alert:
        reasons = []
        if below_f1:
            reasons.append(f"realized F1 {report['f1']:.4f} < floor {config.realized_f1_floor}")
        if below_recall:
            reasons.append(f"realized recall {report['recall']:.4f} < floor {config.realized_recall_floor}")
        emit_alert(
            "decay", "critical",
            "Realized performance dropped below floor: " + "; ".join(reasons),
            {"realized": report, "batch_id": batch_id},
            config,
        )
    return record


# --------------------------------------------------------------------------- #
# The scheduled service wrapper
# --------------------------------------------------------------------------- #
class MonitoringService:
    """Binds a reference distribution to a live-batch provider and runs the
    drift cycle on a cadence.

    `current_batch_provider()` returns `(current_df, current_scores)` for the
    most recent window — in production this reads your scoring log; in the demo
    it samples recent rows. Kept as an injected callable so the schedule loop
    carries no data-access assumptions and stays testable.
    """

    def __init__(self, reference_df, feature_cols, reference_scores,
                 current_batch_provider, config=MONITORING):
        self.reference_df = reference_df
        self.feature_cols = feature_cols
        self.reference_scores = reference_scores
        self.current_batch_provider = current_batch_provider
        self.config = config

    def run_cycle(self) -> dict:
        current_df, current_scores = self.current_batch_provider()
        return run_monitoring_cycle(
            self.reference_df, current_df, self.feature_cols,
            self.reference_scores, current_scores, self.config,
        )

    def serve_forever(self, interval_seconds: int = None):
        """Run run_cycle() every interval. Uses APScheduler's BlockingScheduler
        when installed, else a plain sleep loop. Both call the same run_cycle().
        """
        interval = interval_seconds or self.config.interval_seconds
        logger.info("Starting monitoring service; cadence=%ss", interval)
        try:
            from apscheduler.schedulers.blocking import BlockingScheduler
            sched = BlockingScheduler()
            sched.add_job(self.run_cycle, "interval", seconds=interval,
                          next_run_time=datetime.now())
            sched.start()
        except ImportError:
            import time
            while True:                       # pragma: no cover - trivial loop
                self.run_cycle()
                time.sleep(interval)


# --------------------------------------------------------------------------- #
# __main__ only WIRES the service to a data source; no logic lives here.
# In a real deployment the scheduler would be Airflow/Prefect/Kubeflow or a
# k8s CronJob rather than an in-process loop (noted in src/retraining.py too).
# --------------------------------------------------------------------------- #
def _demo_provider_from_csv(feature_cols):  # pragma: no cover - convenience wiring
    """Build reference + a rolling current-batch provider from the training CSV
    and the current Production model. Used by `python -m src.monitoring_service`.
    """
    from src.data_loader import load_raw, time_based_split
    from src.feature_engineering import build_feature_pipeline
    from src import registry

    bundle = registry.load_bundle()
    df = load_raw()
    train_df, _, test_df = time_based_split(df)
    train_df, _, test_df, _, _ = build_feature_pipeline(train_df, train_df, test_df, persist=False)

    ref = train_df.sample(min(MONITORING.batch_size, len(train_df)), random_state=42)
    ref_scores = bundle["model"].predict_proba(ref[feature_cols])[:, 1]

    def provider():
        cur = test_df.sample(min(MONITORING.batch_size, len(test_df)))
        cur_scores = bundle["model"].predict_proba(cur[feature_cols])[:, 1]
        return cur, cur_scores

    return ref, ref_scores, provider


if __name__ == "__main__":  # pragma: no cover
    from src import registry
    bundle = registry.load_bundle()
    feats = bundle["feature_cols"]
    ref, ref_scores, provider = _demo_provider_from_csv(feats)
    MonitoringService(ref, feats, ref_scores, provider).serve_forever()
