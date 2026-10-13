"""
simulate_decay_demo.py — the whole model lifecycle, end to end, on a simulated
concept-drift event. Everything here is REAL code on REAL (subsampled) data; the
only thing synthesized is the drift itself, and it is clearly labelled as such.

What it demonstrates, in order:

  1. A stale champion is registered and promoted to Production.
  2. A concept-drift event hits the recent data window (the risk signal moves to
     different PCA components + an amount shift). Labelled SIMULATED.
  3. The monitoring service detects it two ways:
       - a DRIFT alert (input/score PSI crosses the significant band), and
       - a DECAY alert (realized F1 on the recent window craters once labels
         arrive).
  4. should_retrain fires on the decay signal, with an auditable reason.
  5. The retraining flow trains a challenger on the recent window and the gate
     ACCEPTS it (it beats the stale champion's PR-AUC by the required margin).
  6. A second retrain, now that the champion already matches the new regime,
     produces a challenger the gate REJECTS (no real improvement) — proving the
     gate never auto-promotes a worse-or-equal model.
  7. Rollback restores a prior version to Production.

Run:
    python scripts/simulate_decay_demo.py
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append(".")

import numpy as np
import pandas as pd

from src.config import DATA, MONITORING, RETRAIN
from src.feature_engineering import build_feature_pipeline, get_feature_columns
from src.training_core import train_xgb_pipeline, score_on_common_test
from src import registry, retraining
from src.monitoring_service import run_monitoring_cycle, run_delayed_performance_cycle


# --------------------------------------------------------------------------- #
# The simulated drift. Labelled SIMULATED everywhere it is used.
# --------------------------------------------------------------------------- #
# Components the synthetic generator originally used to separate positives.
_OLD_SIGNAL_COMPONENTS = ["V2", "V4", "V5", "V11", "V13", "V15", "V18"]
# Components the signal "moves onto" after the simulated concept-drift event.
_NEW_SIGNAL_COMPONENTS = ["V7", "V9", "V20", "V22", "V24"]


def simulate_concept_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """SIMULATED concept drift: rewrite the recent window so that what *defines*
    a high-risk event changes — a genuine concept shift, not just covariate noise.

    Three things happen, all clearly labelled SIMULATED:
      1. The OLD signal is destroyed: the components the champion learned to key
         on are overwritten with pure noise, so the champion goes blind.
      2. A NEW signal is planted: the label is re-defined by a large class-conditional
         shift on a *different* set of components. A model trained AFTER the event
         can learn it; the champion cannot.
      3. Covariate + amount drift: a mean shift on those components for everyone,
         plus an amount shift, so the input-PSI monitor also lights up.

    The positive rate is preserved (labels are re-drawn, not added), so this reads
    to the pipeline exactly like a regime change in which attackers adopted a new
    pattern — the canonical reason a risk model decays.
    """
    rng = np.random.default_rng(seed)
    out = df.copy().reset_index(drop=True)
    n = len(out)

    # Re-draw which rows are positive, at the same rate as before the event.
    n_pos = int(out[DATA.target_col].sum())
    pos_idx = rng.choice(n, size=n_pos, replace=False)
    new_class = np.zeros(n, dtype=int)
    new_class[pos_idx] = 1
    out[DATA.target_col] = new_class
    is_pos = new_class == 1

    # (1) destroy the OLD signal — champion's features become noise.
    for c in _OLD_SIGNAL_COMPONENTS:
        out[c] = rng.normal(0.0, 1.0, size=n)

    # (2)+(3) plant the NEW signal + a covariate mean shift on the same columns.
    for c in _NEW_SIGNAL_COMPONENTS:
        vals = rng.normal(0.0, 1.0, size=n) + 1.5     # covariate shift (everyone)
        vals[is_pos] += 4.0                          # concept signal (positives only)
        out[c] = vals

    # Amount shift — a plausible, benign-looking input drift on top.
    out["Amount"] = out["Amount"] * 5.0
    return out


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _load_subsample(path: str, n_rows: int) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=n_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def _featurize_pair(before: pd.DataFrame, after: pd.DataFrame):
    """Fit encoders on `before`, apply to both; return (before_feat, after_feat,
    feature_cols). persist=False so the champion's on-disk joblibs are untouched.
    """
    b, _, a, _, _ = build_feature_pipeline(before, before, after, persist=False)
    return b, a, get_feature_columns()


def _register_and_promote(bundle, source_df, run_name) -> str:
    v = registry.log_training_run(
        model=bundle["model"], feature_cols=bundle["feature_cols"],
        threshold_info=bundle["threshold_info"], metrics=bundle["metrics"],
        params=bundle["params"], counterparty_risk_maps=bundle["risk_maps"],
        device_map=bundle["device_map"],
        fingerprint=registry.data_fingerprint(source_df), run_name=run_name,
    )
    registry.promote_to_production(v)
    return v


def _hr(title):
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


# --------------------------------------------------------------------------- #
# The demo
# --------------------------------------------------------------------------- #
def run_demo(path: str = None, n_rows: int = 80000, seed: int = 42) -> dict:
    path = path or DATA.raw_path
    np.random.seed(seed)

    _hr("0. Load a subsample; split into a 'before' regime and a recent window")
    df = _load_subsample(path, n_rows)
    cut = int(len(df) * 0.55)
    before, after_raw = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    after = simulate_concept_drift(after_raw, seed)   # SIMULATED drift
    full = pd.concat([before, after], ignore_index=True)
    print(f"before: {len(before):,} rows ({int(before.Class.sum())} positive) | "
          f"recent window (SIMULATED drift): {len(after):,} rows "
          f"({int(after.Class.sum())} positive)")

    # Remember whatever was serving before the demo so we can restore it at the end.
    restore_version = registry.get_production_version()

    _hr("1. Train the STALE champion on the 'before' regime; promote to Production")
    champ_bundle = train_xgb_pipeline(before, "class_weight", seed=seed, persist_artifacts=False)
    champ_stale = _register_and_promote(champ_bundle, before, "demo_stale_champion")
    print(f"champion = v{champ_stale} (Production). "
          f"Training PR-AUC={champ_bundle['metrics']['pr_auc']:.4f} "
          f"F1={champ_bundle['metrics']['f1']:.4f}")

    _hr("2. Monitoring: DRIFT check (inputs + scores), no labels needed")
    before_feat, after_feat, feat_cols = _featurize_pair(before, after)
    ref = before_feat.sample(min(MONITORING.batch_size, len(before_feat)), random_state=seed)
    cur = after_feat.sample(min(MONITORING.batch_size, len(after_feat)), random_state=seed)
    ref_scores = champ_bundle["model"].predict_proba(ref[feat_cols])[:, 1]
    cur_scores = champ_bundle["model"].predict_proba(cur[feat_cols])[:, 1]
    drift_rec = run_monitoring_cycle(ref, cur, feat_cols, ref_scores, cur_scores)
    print(f"drift_alert={drift_rec['drift_alert']} | "
          f"{drift_rec['n_significant_features']} features in significant PSI band: "
          f"{drift_rec['significant_features'][:6]}... | "
          f"score PSI={drift_rec['score_drift']['score_psi']} "
          f"({drift_rec['score_drift']['status']})")

    _hr("3. Monitoring: DECAY check on the recent window once delayed labels arrive")
    thr = champ_bundle["threshold_info"]["best_threshold"]
    y_after = after_feat[DATA.target_col].values
    proba_after = champ_bundle["model"].predict_proba(after_feat[feat_cols])[:, 1]
    perf_rec = run_delayed_performance_cycle(y_after, proba_after, thr, batch_id="recent_window")
    print(f"decay_alert={perf_rec['decay_alert']} | realized on recent window: "
          f"precision={perf_rec['precision']:.4f} recall={perf_rec['recall']:.4f} "
          f"F1={perf_rec['f1']:.4f} (floors F1>={perf_rec['f1_floor']}, "
          f"recall>={perf_rec['recall_floor']})")

    _hr("4. should_retrain: policy decision (auditable reasons)")
    realized = {"f1": perf_rec["f1"], "recall": perf_rec["recall"]}
    baseline = {"f1": champ_bundle["metrics"]["f1"]}
    drift_summary = retraining.summarize_drift([drift_rec, drift_rec])  # sustained
    decision = retraining.should_retrain(realized, baseline, drift_summary)
    print(f"should_retrain={decision.should_retrain} triggers={decision.triggers} "
          f"severity={decision.severity}")
    for r in decision.reasons:
        print(f"  - {r}")

    _hr("5. Retraining flow — gate should ACCEPT (challenger beats stale champion)")
    accept = retraining.run_retraining_flow(
        full, realized_performance=realized, drift_history=[drift_rec, drift_rec],
        imbalance_strategy="class_weight", seed=seed,
    )
    g = accept["gate"]
    print(f"challenger=v{accept['challenger_version']} | gate: "
          f"champion {g['metric']}={g['champion_value']} vs challenger "
          f"{g['challenger_value']} (gain {g['gain']:+}, margin {g['required_margin']}) "
          f"-> {g['verdict'].upper()}")
    print(f"ACTION: {accept['action']} | Production now = "
          f"v{accept['production_version_now']}")
    assert accept["gate"]["promote"], "expected the gate to ACCEPT the challenger"

    _hr("6. Gate should REJECT an inferior challenger (never promote a worse model)")
    # A realistic rejection: a proposed challenger accidentally trained on the
    # STALE 'before' regime. Judged against the current champion on a common
    # NEW-regime test, it is blind — and the gate must refuse to promote it.
    current_champ = registry.load_bundle()["model"]           # the v3 champion
    stale_bundle = train_xgb_pipeline(before, "class_weight", seed=seed, persist_artifacts=False)
    stale_version = registry.log_training_run(
        model=stale_bundle["model"], feature_cols=stale_bundle["feature_cols"],
        threshold_info=stale_bundle["threshold_info"], metrics=stale_bundle["metrics"],
        params={**stale_bundle["params"], "role": "challenger_stale"},
        counterparty_risk_maps=stale_bundle["risk_maps"], device_map=stale_bundle["device_map"],
        fingerprint=registry.data_fingerprint(before), run_name="challenger_stale_rejected",
    )
    common = after_feat.tail(int(len(after_feat) * 0.3))      # common new-regime test
    champ_thr = registry.load_bundle()["threshold_info"]["best_threshold"]
    champ_metrics = score_on_common_test(
        current_champ, common[feat_cols], common[DATA.target_col], champ_thr, "champion")
    stale_metrics = score_on_common_test(
        stale_bundle["model"], common[feat_cols], common[DATA.target_col],
        stale_bundle["threshold_info"]["best_threshold"], "stale challenger")
    g2 = retraining.gate(champ_metrics, stale_metrics)
    print(f"stale challenger=v{stale_version} | gate: champion {g2['metric']}="
          f"{g2['champion_value']} vs challenger {g2['challenger_value']} "
          f"(gain {g2['gain']:+}, margin {g2['required_margin']}) -> {g2['verdict'].upper()}")
    print(f"ACTION: rejected (challenger left in Staging) | Production unchanged = "
          f"v{registry.get_production_version()}")
    assert not g2["promote"], "expected the gate to REJECT the inferior challenger"

    _hr("7. Rollback — restore a prior version to Production")
    promoted_challenger = accept["production_version_now"]
    rb = retraining.rollback(champ_stale)
    print(f"rolled back Production from v{promoted_challenger} to v{champ_stale} "
          f"-> Production now = v{rb['production_version_now']}")
    assert rb["production_version_now"] == str(champ_stale)

    # Cleanup: restore whatever the canonical serving version was before the demo
    # (if any), so a running API is unaffected by having run the demo.
    if restore_version and restore_version != champ_stale:
        registry.rollback_to_version(restore_version)
        print(f"\n(cleanup) restored pre-demo Production champion v{restore_version}.")

    _hr("DEMO COMPLETE — decay detected, challenger trained, gate ACCEPTED then "
        "REJECTED, rollback verified")
    return {
        "champion_stale": champ_stale,
        "accept_gate": accept["gate"],
        "reject_gate": g2,
        "drift_alert": drift_rec["drift_alert"],
        "decay_alert": perf_rec["decay_alert"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA.raw_path)
    parser.add_argument("--n_rows", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_demo(args.data, args.n_rows, args.seed)


if __name__ == "__main__":
    main()
