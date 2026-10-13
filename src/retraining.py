"""
retraining.py — the automated retraining trigger and the champion/challenger
promotion gate. This is where a decay signal becomes a decision.

The pieces, in the order the flow uses them:

  1. should_retrain(...)  — an explicit, config-driven, AUDITABLE policy. It
     returns *why* it decided, not just a bool. Two triggers, kept distinct:
       - PRIMARY  (decay): realized performance fell below a floor, or dropped
         sharply versus the champion's training baseline. The model actually
         got worse — retrain now.
       - SECONDARY (drift): significant input/score drift that has PERSISTED for
         several cycles. An early warning that may warrant a pre-emptive retrain
         before labels confirm decay. A single spike is investigated, not acted on.

  2. train_challenger(...) — retrain on the NEWER data window (the recent regime
     that triggered the alarm), via the exact same leakage-safe pipeline as the
     champion (src/training_core.py). Registered as a new Staging version.

  3. gate(...) — promote challenger -> Production ONLY if it beats the champion
     on the primary metric (PR-AUC) by a configurable margin, judged on a common
     chronologically-held-out test split. Otherwise keep the champion and log the
     rejected challenger. This gate is the whole point: never auto-promote a
     worse model.

  4. rollback(...) — promote a specified prior version back to Production, for
     when a promoted model misbehaves.

  run_retraining_flow(...) orchestrates monitor -> detect -> retrain -> evaluate
  -> gated promote/reject -> record. It is a plain Python orchestrator on
  purpose; in a real deployment the DAG (schedule, retries, backfill, lineage)
  would live in Airflow / Prefect / Kubeflow Pipelines and each numbered step
  would be a task. The decision logic below would be unchanged — that separation
  is deliberate.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

from src.config import RETRAIN, REGISTRY, DATA
from src import registry
from src.training_core import train_xgb_pipeline, score_on_common_test


# --------------------------------------------------------------------------- #
# The decision object — auditable by construction
# --------------------------------------------------------------------------- #
@dataclass
class RetrainDecision:
    should_retrain: bool
    triggers: list = field(default_factory=list)   # subset of {"decay", "drift"}
    reasons: list = field(default_factory=list)     # human-readable justification
    severity: str = "none"                          # "critical" | "warning" | "none"

    def to_dict(self) -> dict:
        return asdict(self)


def should_retrain(realized_performance: dict, baseline_performance: dict,
                   drift_summary: dict = None, config=RETRAIN) -> RetrainDecision:
    """Decide whether to retrain, and record why.

    Args:
      realized_performance: realized metrics from delayed labels, e.g.
        {"f1": .., "recall": ..}. None if no labels have arrived yet — then only
        the drift trigger can fire.
      baseline_performance: metrics recorded at the champion's training time,
        e.g. {"f1": ..} (from the registry). Used for the relative-drop test.
      drift_summary: {"consecutive_significant_cycles": int,
                      "n_significant_features": int}. None if drift not evaluated.

    Returns a RetrainDecision carrying the trigger(s), reasons, and severity.
    """
    triggers, reasons = [], []

    # ---- PRIMARY: realized-performance decay (concept drift) ----
    if realized_performance is not None and "f1" in realized_performance:
        f1 = realized_performance["f1"]
        if f1 < config.decay_f1_abs_floor:
            triggers.append("decay")
            reasons.append(
                f"realized F1 {f1:.4f} below absolute floor {config.decay_f1_abs_floor}"
            )
        base_f1 = (baseline_performance or {}).get("f1")
        if base_f1:
            drop = (base_f1 - f1) / base_f1
            if drop > config.decay_f1_relative_drop:
                if "decay" not in triggers:
                    triggers.append("decay")
                reasons.append(
                    f"realized F1 {f1:.4f} is {drop:.0%} below training baseline "
                    f"{base_f1:.4f} (> {config.decay_f1_relative_drop:.0%} allowed)"
                )

    # ---- SECONDARY: sustained input/score drift (early trigger) ----
    if drift_summary:
        cyc = drift_summary.get("consecutive_significant_cycles", 0)
        nfeat = drift_summary.get("n_significant_features", 0)
        if cyc >= config.drift_sustained_cycles and nfeat >= config.drift_min_significant_features:
            triggers.append("drift")
            reasons.append(
                f"significant drift sustained {cyc} consecutive cycles "
                f"(>= {config.drift_sustained_cycles}), {nfeat} features shifted "
                f"(>= {config.drift_min_significant_features})"
            )

    should = len(triggers) > 0
    severity = "critical" if "decay" in triggers else ("warning" if should else "none")
    if not should:
        reasons.append("all monitored signals within configured bands")
    return RetrainDecision(should, triggers, reasons, severity)


def summarize_drift(drift_history: list, config=RETRAIN) -> dict:
    """Collapse a drift-history list (from monitoring_service) into the small
    summary should_retrain needs: how many of the most-recent consecutive cycles
    fired a drift alert, and the latest significant-feature count.
    """
    if not drift_history:
        return {"consecutive_significant_cycles": 0, "n_significant_features": 0}
    consecutive = 0
    for rec in reversed(drift_history):
        if rec.get("drift_alert"):
            consecutive += 1
        else:
            break
    return {
        "consecutive_significant_cycles": consecutive,
        "n_significant_features": drift_history[-1].get("n_significant_features", 0),
    }


# --------------------------------------------------------------------------- #
# Challenger training
# --------------------------------------------------------------------------- #
def _newer_window(df, window_frac: float):
    """The chronologically-newest `window_frac` of the data — the recent regime
    a challenger should learn. df is assumed already sorted by time (load_raw
    guarantees this); we re-sort defensively.
    """
    df = df.sort_values(DATA.time_col).reset_index(drop=True)
    start = int(len(df) * (1.0 - window_frac))
    return df.iloc[start:].reset_index(drop=True)


def train_challenger(df, imbalance_strategy: str = "class_weight", seed: int = None,
                     config=RETRAIN, register: bool = True) -> dict:
    """Train a challenger on the newer data window and (optionally) register it
    as a new Staging version. Returns {"version", "bundle", "fingerprint"}.

    The challenger is trained by train_xgb_pipeline — the identical pipeline that
    produced the champion — so a later gate comparison reflects DATA/regime
    differences, not procedural ones. persist_artifacts=False so it never
    clobbers the champion's on-disk joblibs.
    """
    seed = DATA.random_state if seed is None else seed
    window_df = _newer_window(df, config.challenger_window_frac)
    bundle = train_xgb_pipeline(window_df, imbalance_strategy, seed=seed,
                                persist_artifacts=False)
    fingerprint = registry.data_fingerprint(window_df)

    version = None
    if register:
        version = registry.log_training_run(
            model=bundle["model"],
            feature_cols=bundle["feature_cols"],
            threshold_info=bundle["threshold_info"],
            metrics=bundle["metrics"],
            params={**bundle["params"], "role": "challenger",
                    "window_frac": config.challenger_window_frac},
            counterparty_risk_maps=bundle["risk_maps"],
            device_map=bundle["device_map"],
            fingerprint=fingerprint,
            stage=REGISTRY.staging_stage,
            run_name="challenger_retrain",
        )
    return {"version": version, "bundle": bundle, "fingerprint": fingerprint}


# --------------------------------------------------------------------------- #
# Champion vs challenger evaluation + the gate
# --------------------------------------------------------------------------- #
def evaluate_champion_vs_challenger(challenger_bundle: dict, champion_model=None,
                                    champion_threshold: float = 0.5) -> dict:
    """Score champion and challenger on a COMMON held-out test split — the
    challenger window's chronologically-last slice — so both are judged on
    identical rows.

    Note (honesty): the common test features are encoded with the challenger's
    encoders (counterparty-risk map fit on the challenger window). Both models see
    the SAME matrix, so the comparison between them is fair; it is not a
    from-scratch re-featurization for the champion. Good enough to rank two
    models; flagged rather than hidden.
    """
    X_test = challenger_bundle["X_test"]
    y_test = challenger_bundle["y_test"]

    challenger_metrics = challenger_bundle["metrics"]  # already on this X_test
    if champion_model is None:
        champion_model = registry.load_bundle(REGISTRY.production_stage)["model"]
    champion_metrics = score_on_common_test(
        champion_model, X_test, y_test, threshold=champion_threshold,
        model_name="champion (on challenger test)",
    )
    return {"champion": champion_metrics, "challenger": challenger_metrics}


def gate(champion_metrics: dict, challenger_metrics: dict, config=RETRAIN) -> dict:
    """The promotion gate. Promote the challenger only if it beats the champion
    on the primary metric (PR-AUC) by at least the configured margin.

    Returns a fully spelled-out decision so the outcome is auditable: both
    values, the gain, the required margin, and the boolean.
    """
    metric = config.primary_metric
    champ = float(champion_metrics.get(metric, 0.0))
    chall = float(challenger_metrics.get(metric, 0.0))
    gain = chall - champ
    promote = gain >= config.promotion_min_pr_auc_gain
    return {
        "metric": metric,
        "champion_value": round(champ, 6),
        "challenger_value": round(chall, 6),
        "gain": round(gain, 6),
        "required_margin": config.promotion_min_pr_auc_gain,
        "promote": bool(promote),
        "verdict": "promote_challenger" if promote else "keep_champion",
    }


def rollback(version, config=RETRAIN) -> dict:
    """Roll Production back to a specified prior version and record it."""
    registry.rollback_to_version(version)
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "rollback",
        "rolled_back_to_version": str(version),
        "production_version_now": registry.get_production_version(),
    }
    _record_decision(rec, config)
    return rec


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _record_decision(record: dict, config=RETRAIN) -> str:
    os.makedirs(os.path.dirname(config.decisions_log_path), exist_ok=True)
    with open(config.decisions_log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return config.decisions_log_path


def run_retraining_flow(df, realized_performance: dict = None,
                        drift_history: list = None, imbalance_strategy: str = "class_weight",
                        seed: int = None, force: bool = False, config=RETRAIN) -> dict:
    """Orchestrate the full loop and record the decision.

    monitor(inputs already gathered) -> should_retrain -> train challenger ->
    evaluate vs champion on a common test -> gated promote/reject -> record.

    In production this function's body would be a DAG (Airflow/Prefect/Kubeflow):
    each step a retryable, observable task with data lineage. Here it is one
    process so the whole lifecycle is demonstrable locally end-to-end.
    """
    prod_version = registry.get_production_version()
    champion_bundle = registry.load_bundle(REGISTRY.production_stage)
    baseline_performance = {
        "f1": champion_bundle["metrics"].get("f1"),
        "pr_auc": champion_bundle["metrics"].get("pr_auc"),
    }
    drift_summary = summarize_drift(drift_history, config) if drift_history is not None else None

    decision = should_retrain(realized_performance, baseline_performance, drift_summary, config)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "champion_version": prod_version,
        "baseline_performance": baseline_performance,
        "realized_performance": realized_performance,
        "drift_summary": drift_summary,
        "decision": decision.to_dict(),
    }

    if not decision.should_retrain and not force:
        result["action"] = "no_retrain"
        _record_decision(result, config)
        return result

    # ---- retrain challenger on the recent window ----
    challenger = train_challenger(df, imbalance_strategy, seed=seed, config=config)
    result["challenger_version"] = challenger["version"]
    result["challenger_fingerprint"] = challenger["fingerprint"]

    # ---- evaluate champion vs challenger on a common test split ----
    champion_threshold = champion_bundle["threshold_info"].get("best_threshold", 0.5)
    comparison = evaluate_champion_vs_challenger(
        challenger["bundle"], champion_model=champion_bundle["model"],
        champion_threshold=champion_threshold,
    )
    result["comparison"] = comparison

    # ---- the gate ----
    gate_result = gate(comparison["champion"], comparison["challenger"], config)
    result["gate"] = gate_result

    if gate_result["promote"]:
        registry.promote_to_production(challenger["version"])
        result["action"] = "promoted_challenger"
        result["production_version_now"] = registry.get_production_version()
    else:
        # Challenger stays in Staging (rejected); champion keeps serving.
        result["action"] = "rejected_challenger"
        result["production_version_now"] = prod_version

    _record_decision(result, config)
    return result
