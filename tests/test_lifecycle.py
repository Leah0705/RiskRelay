"""
test_lifecycle.py — tests for the model-lifecycle machinery added on top of the
training pipeline: the retraining trigger, the champion/challenger gate, the
registry's promote/rollback, and the monitoring service's persistence + alerts.

These are deliberately fast and hermetic: the registry test spins up a throwaway
MLflow file store in a tmp dir and trains tiny XGBoost models, so nothing touches
the real ./mlruns. The trigger and gate tests are pure-function tests.
"""

import sys
import numpy as np
import pandas as pd
import pytest

sys.path.append(".")

from src.config import RETRAIN, MONITORING, REGISTRY
from src import retraining
from src.retraining import should_retrain, gate, summarize_drift
from src import registry
from src import monitoring_service
from src.monitoring_service import run_monitoring_cycle, run_delayed_performance_cycle


# --------------------------------------------------------------------------- #
# should_retrain — stable / drift-only / decay
# --------------------------------------------------------------------------- #
def test_should_retrain_stable_does_not_retrain():
    realized = {"f1": 0.20, "recall": 0.60}          # healthy vs baseline
    baseline = {"f1": 0.22}
    drift = {"consecutive_significant_cycles": 0, "n_significant_features": 0}
    d = should_retrain(realized, baseline, drift)
    assert d.should_retrain is False
    assert d.triggers == []
    assert d.severity == "none"


def test_should_retrain_decay_triggers_primary():
    realized = {"f1": 0.01, "recall": 0.02}          # collapsed
    baseline = {"f1": 0.20}
    drift = {"consecutive_significant_cycles": 0, "n_significant_features": 0}
    d = should_retrain(realized, baseline, drift)
    assert d.should_retrain is True
    assert "decay" in d.triggers
    assert d.severity == "critical"
    assert any("F1" in r for r in d.reasons)          # reason is auditable


def test_should_retrain_relative_drop_triggers_even_above_floor():
    # F1 above the absolute floor, but a large relative drop vs baseline.
    realized = {"f1": 0.10, "recall": 0.40}
    baseline = {"f1": 0.20}                            # 50% drop > 25% allowed
    d = should_retrain(realized, baseline, drift_summary=None)
    assert d.should_retrain is True
    assert "decay" in d.triggers


def test_should_retrain_drift_only_is_secondary_trigger():
    realized = {"f1": 0.20, "recall": 0.60}           # performance fine...
    baseline = {"f1": 0.20}
    drift = {"consecutive_significant_cycles": RETRAIN.drift_sustained_cycles,
             "n_significant_features": RETRAIN.drift_min_significant_features}
    d = should_retrain(realized, baseline, drift)
    assert d.should_retrain is True
    assert d.triggers == ["drift"]                    # ...but sustained drift alone triggers
    assert d.severity == "warning"


def test_should_retrain_single_drift_spike_does_not_trigger():
    drift = {"consecutive_significant_cycles": 1,     # below sustained threshold
             "n_significant_features": 10}
    d = should_retrain({"f1": 0.2}, {"f1": 0.2}, drift)
    assert d.should_retrain is False


def test_summarize_drift_counts_trailing_consecutive_alerts():
    history = [
        {"drift_alert": True, "n_significant_features": 4},
        {"drift_alert": False, "n_significant_features": 0},
        {"drift_alert": True, "n_significant_features": 5},
        {"drift_alert": True, "n_significant_features": 6},
    ]
    s = summarize_drift(history)
    assert s["consecutive_significant_cycles"] == 2   # only the trailing run
    assert s["n_significant_features"] == 6


# --------------------------------------------------------------------------- #
# The gate — must promote on improvement and reject on regression
# --------------------------------------------------------------------------- #
def test_gate_promotes_on_sufficient_improvement():
    champ = {"pr_auc": 0.10}
    chall = {"pr_auc": 0.10 + RETRAIN.promotion_min_pr_auc_gain + 0.01}
    g = gate(champ, chall)
    assert g["promote"] is True
    assert g["verdict"] == "promote_challenger"
    assert g["gain"] > g["required_margin"]


def test_gate_rejects_on_regression():
    champ = {"pr_auc": 0.15}
    chall = {"pr_auc": 0.12}                          # worse
    g = gate(champ, chall)
    assert g["promote"] is False
    assert g["verdict"] == "keep_champion"


def test_gate_rejects_on_insufficient_margin():
    # Better, but not by the required margin — still rejected.
    champ = {"pr_auc": 0.10}
    chall = {"pr_auc": 0.10 + RETRAIN.promotion_min_pr_auc_gain / 2.0}
    g = gate(champ, chall)
    assert g["promote"] is False


# --------------------------------------------------------------------------- #
# Registry — promote + rollback change the resolved Production version
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_registry(tmp_path, monkeypatch):
    """Point the registry at a throwaway file store + unique model name so the
    real ./mlruns is never touched.
    """
    monkeypatch.setattr(REGISTRY, "tracking_uri", f"file:{tmp_path/'mlruns'}")
    monkeypatch.setattr(REGISTRY, "registered_model_name", "risk_xgb_test")
    yield


def _tiny_bundle(seed):
    from xgboost import XGBClassifier
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(200, 4)), columns=list("abcd"))
    y = (X["a"] + rng.normal(0, 0.5, size=200) > 0).astype(int)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=seed).fit(X, y)
    return dict(
        model=model, feature_cols=list("abcd"),
        threshold_info={"best_threshold": 0.5},
        metrics={"pr_auc": 0.5, "f1": 0.5, "roc_auc": 0.5, "precision": 0.5, "recall": 0.5},
        params={"seed": seed}, risk_maps={"global_mean": 0.01}, device_map={},
    )


def _register(bundle):
    return registry.log_training_run(
        model=bundle["model"], feature_cols=bundle["feature_cols"],
        threshold_info=bundle["threshold_info"], metrics=bundle["metrics"],
        params=bundle["params"], counterparty_risk_maps=bundle["risk_maps"],
        device_map=bundle["device_map"],
        fingerprint={"n_rows": 200, "content_sha256": "x", "date_range": [None, None]},
    )


def test_registry_promote_and_rollback_change_production(isolated_registry):
    v1 = _register(_tiny_bundle(1))
    v2 = _register(_tiny_bundle(2))

    registry.promote_to_production(v1)
    assert registry.get_production_version() == v1

    # Promote v2: it becomes Production, v1 is archived.
    registry.promote_to_production(v2)
    assert registry.get_production_version() == v2

    # Rollback to v1: Production resolves back to v1.
    registry.rollback_to_version(v1)
    assert registry.get_production_version() == v1

    # The bundle we load back for serving is the rolled-back version.
    bundle = registry.load_bundle(REGISTRY.production_stage)
    assert bundle["version"] == v1
    assert bundle["feature_cols"] == list("abcd")


# --------------------------------------------------------------------------- #
# Monitoring service — persists a record and alerts on a shifted distribution
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_monitoring(tmp_path, monkeypatch):
    monkeypatch.setattr(MONITORING, "history_dir", str(tmp_path / "monitoring"))
    yield


def _drift_frame(n, shift, seed):
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(5)]
    data = {c: rng.normal(shift, 1.0, size=n) for c in cols}
    return pd.DataFrame(data), cols


def test_monitoring_cycle_persists_and_alerts_on_shift(isolated_monitoring):
    ref, cols = _drift_frame(4000, shift=0.0, seed=1)
    cur, _ = _drift_frame(4000, shift=3.0, seed=2)      # big shift on every feature
    ref_scores = np.random.default_rng(3).uniform(0, 0.2, size=4000)
    cur_scores = np.random.default_rng(4).uniform(0.6, 1.0, size=4000)  # shifted scores

    rec = run_monitoring_cycle(ref, cur, cols, ref_scores, cur_scores)

    assert rec["drift_alert"] is True
    assert rec["n_significant_features"] >= MONITORING.min_significant_features
    assert rec["score_drift"]["status"] == "significant_shift"

    # A timestamped record was persisted to the drift history.
    history = monitoring_service.load_history(monitoring_service.DRIFT_HISTORY)
    assert len(history) == 1
    assert history[0]["drift_alert"] is True


def test_monitoring_cycle_stable_when_no_shift(isolated_monitoring):
    ref, cols = _drift_frame(4000, shift=0.0, seed=1)
    cur, _ = _drift_frame(4000, shift=0.0, seed=5)      # same distribution
    rec = run_monitoring_cycle(ref, cur, cols)
    assert rec["drift_alert"] is False
    assert rec["n_significant_features"] == 0


def test_delayed_performance_cycle_alerts_on_decay(isolated_monitoring):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    # Predictions uncorrelated with truth -> low realized F1/recall -> decay alert.
    y_proba = rng.uniform(0, 0.4, size=500)
    rec = run_delayed_performance_cycle(y_true, y_proba, threshold=0.5, batch_id="b1")
    assert rec["decay_alert"] is True
    history = monitoring_service.load_history(monitoring_service.PERFORMANCE_HISTORY)
    assert len(history) == 1 and history[0]["batch_id"] == "b1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
