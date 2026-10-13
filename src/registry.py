"""
registry.py — the model registry, backed by a LOCAL MLflow file store.

Why a registry at all: without one, "which model is in production?" is answered
by a mutable file path (models/xgb_risk_model.joblib) with no record of the
data, config, metrics, or threshold that produced it. That is fine until the
day a model misbehaves and you need to answer "what changed, and what do we roll
back to?" — at which point a versioned registry is the difference between a
one-line rollback and an archaeology project.

Design choices, kept deliberately small and local:
  * Backend is MLflow's file store (`file:./mlruns`) — no server, no database,
    no cloud. `mlflow ui` can still browse it after the fact.
  * We use MLflow's two canonical registry STAGES, "Staging" and "Production",
    rather than hand-rolling our own. Promotion = transition a version to
    Production and archive the incumbent; rollback = promote an older version
    back. (MLflow 3 replaces stages with aliases; we pin MLflow 2.x so the
    stage vocabulary the rest of this codebase speaks stays first-class.)
  * Every registered version carries, as logged artifacts, the exact threshold,
    feature list, and encoders needed to SERVE it, plus a fingerprint of the
    training data — so any production model traces back to the precise rows it
    was trained on.

The registry, not the joblib files, is the source of truth for serving.
"""

import hashlib
import json
import os
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from src.config import REGISTRY, DATA, FEATURES

# MLflow 2.x emits FutureWarnings about stages being superseded by aliases in
# 3.x. We intentionally use stages (see module docstring); silence the noise so
# the training logs stay readable.
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# Sub-path under each run where we stash the serving sidecar artifacts.
_SERVING_DIR = "serving"


def _init() -> MlflowClient:
    """Point MLflow at the local file store and ensure the experiment exists.
    Idempotent — safe to call on every entry point.
    """
    mlflow.set_tracking_uri(REGISTRY.tracking_uri)
    mlflow.set_experiment(REGISTRY.experiment_name)
    return MlflowClient()


# --------------------------------------------------------------------------- #
# Data fingerprint
# --------------------------------------------------------------------------- #
def data_fingerprint(df: pd.DataFrame) -> dict:
    """A compact, deterministic fingerprint of a training dataset so a
    registered model can be traced back to the *exact* data it saw.

    Captures three independent facts:
      - row count,
      - a content hash: sha256 over a sorted, stable per-row key (so the same
        rows in any order fingerprint identically, but a single changed/added
        row does not),
      - the date range of the window.

    The per-row key prefers an explicit event_id; otherwise it is built
    from the identity + amount + timestamp columns that uniquely pin a row in
    this schema.
    """
    if "event_id" in df.columns:
        keys = df["event_id"].astype(str)
    else:
        parts = []
        for col in [DATA.time_col, FEATURES.account_col, FEATURES.counterparty_col, FEATURES.amount_col]:
            if col in df.columns:
                parts.append(df[col].astype(str))
        keys = parts[0].str.cat(parts[1:], sep="|") if parts else df.index.to_series().astype(str)

    digest = hashlib.sha256("\n".join(sorted(keys.tolist())).encode()).hexdigest()

    date_min = date_max = None
    if DATA.time_col in df.columns:
        ts = pd.to_datetime(df[DATA.time_col])
        date_min, date_max = str(ts.min()), str(ts.max())

    return {
        "n_rows": int(len(df)),
        "content_sha256": digest,
        "date_range": [date_min, date_max],
    }


# --------------------------------------------------------------------------- #
# Logging + registration
# --------------------------------------------------------------------------- #
def log_training_run(
    *,
    model,
    feature_cols: list,
    threshold_info: dict,
    metrics: dict,
    params: dict,
    counterparty_risk_maps: dict,
    device_map: dict,
    fingerprint: dict,
    stage: str = None,
    run_name: str = None,
) -> str:
    """Log one training run and register its model version.

    Logs params, held-out test metrics, the tuned threshold, the feature list,
    the counterparty-risk encoders, the device map, and the data fingerprint, then
    registers the fitted model under REGISTRY.registered_model_name and moves
    the new version to `stage` (default: Staging).

    Returns the new version number (as a string, MLflow's native type).
    """
    client = _init()
    stage = stage or REGISTRY.staging_stage

    with mlflow.start_run(run_name=run_name) as run:
        # Flat, queryable params.
        mlflow.log_params(_flatten_params(params))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("tuned_threshold", threshold_info.get("best_threshold"))

        # Held-out test metrics (the numbers the gate later compares on).
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})

        # Data lineage — fingerprint of the exact training rows.
        mlflow.log_dict(fingerprint, f"{_SERVING_DIR}/data_fingerprint.json")

        # Serving sidecars: everything api.py needs beyond the model itself.
        mlflow.log_dict({"feature_cols": feature_cols}, f"{_SERVING_DIR}/feature_list.json")
        mlflow.log_dict(threshold_info, f"{_SERVING_DIR}/threshold.json")
        _log_joblib(counterparty_risk_maps, f"{_SERVING_DIR}/counterparty_risk_map.joblib")
        _log_joblib(device_map, f"{_SERVING_DIR}/device_map.joblib")

        # The model itself, registered in the model registry.
        model_info = mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=REGISTRY.registered_model_name,
        )
        version = model_info.registered_model_version

    # New versions land in Staging by default; promotion to Production is a
    # separate, gated decision (see src/retraining.py).
    client.transition_model_version_stage(
        REGISTRY.registered_model_name, version, stage
    )
    return str(version)


def _flatten_params(params: dict) -> dict:
    """MLflow params must be scalar-ish; flatten nested dicts (e.g. xgb params)
    into dotted keys and stringify lists.
    """
    flat = {}
    for k, v in params.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        elif isinstance(v, (list, tuple)):
            flat[k] = ",".join(map(str, v))
        else:
            flat[k] = v
    return flat


def _log_joblib(obj, artifact_path: str):
    """Log an arbitrary python object as a joblib artifact under the run."""
    subdir, fname = os.path.split(artifact_path)
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, fname)
        joblib.dump(obj, local)
        mlflow.log_artifact(local, artifact_path=subdir or None)


# --------------------------------------------------------------------------- #
# Stage transitions: promote / rollback
# --------------------------------------------------------------------------- #
def promote_to_production(version) -> None:
    """Promote a version to Production and archive whatever was there before.

    `archive_existing_versions=True` guarantees there is only ever one
    Production version — the champion — so "the model we serve" is never
    ambiguous.
    """
    client = _init()
    client.transition_model_version_stage(
        REGISTRY.registered_model_name,
        str(version),
        REGISTRY.production_stage,
        archive_existing_versions=True,
    )


def rollback_to_version(version) -> None:
    """Roll back Production to a specified prior version.

    Mechanically identical to a promotion — that is the point: a rollback is not
    a special code path, it is just promoting a known-good older version back to
    Production. Use it when a freshly promoted model misbehaves in production.
    """
    promote_to_production(version)


def stage_version(version, stage: str) -> None:
    """Move a version to an arbitrary stage (used by tests and tooling)."""
    client = _init()
    client.transition_model_version_stage(
        REGISTRY.registered_model_name, str(version), stage
    )


# --------------------------------------------------------------------------- #
# Resolution / loading
# --------------------------------------------------------------------------- #
def get_current_version(stage: str) -> str:
    """Return the (single, latest) version currently in `stage`, or None."""
    client = _init()
    versions = client.get_latest_versions(REGISTRY.registered_model_name, stages=[stage])
    return str(versions[0].version) if versions else None


def get_production_version() -> str:
    return get_current_version(REGISTRY.production_stage)


def list_versions() -> list:
    """All registered versions with their stage and key metrics, newest first."""
    client = _init()
    out = []
    for mv in client.search_model_versions(f"name = '{REGISTRY.registered_model_name}'"):
        run = client.get_run(mv.run_id) if mv.run_id else None
        out.append({
            "version": str(mv.version),
            "stage": mv.current_stage,
            "run_id": mv.run_id,
            "pr_auc": (run.data.metrics.get("pr_auc") if run else None),
        })
    out.sort(key=lambda r: int(r["version"]), reverse=True)
    return out


def _download(run_id: str, artifact_path: str) -> str:
    return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)


def load_bundle(stage: str = None, version=None) -> dict:
    """Load a complete, servable bundle for a stage (default Production) or a
    specific version: the model plus every sidecar the API needs — threshold,
    feature list, counterparty-risk encoders, device map — and the version metadata.

    This is what api.py calls at startup. It resolves BY STAGE, never by a
    hardcoded artifact path, so promoting a new Production version and
    restarting the service is all it takes to ship a new model.
    """
    client = _init()
    stage = stage or REGISTRY.production_stage

    if version is None:
        version = get_current_version(stage)
        if version is None:
            raise RuntimeError(
                f"No model version in stage '{stage}' for "
                f"'{REGISTRY.registered_model_name}'. Train and register one first "
                f"(python train.py)."
            )

    mv = client.get_model_version(REGISTRY.registered_model_name, str(version))
    run_id = mv.run_id

    model = mlflow.xgboost.load_model(
        f"models:/{REGISTRY.registered_model_name}/{version}"
    )
    feature_cols = json.load(open(_download(run_id, f"{_SERVING_DIR}/feature_list.json")))["feature_cols"]
    threshold_info = json.load(open(_download(run_id, f"{_SERVING_DIR}/threshold.json")))
    counterparty_risk_maps = joblib.load(_download(run_id, f"{_SERVING_DIR}/counterparty_risk_map.joblib"))
    device_map = joblib.load(_download(run_id, f"{_SERVING_DIR}/device_map.joblib"))
    fingerprint = json.load(open(_download(run_id, f"{_SERVING_DIR}/data_fingerprint.json")))

    return {
        "model": model,
        "feature_cols": feature_cols,
        "threshold_info": threshold_info,
        "counterparty_risk_maps": counterparty_risk_maps,
        "device_map": device_map,
        "fingerprint": fingerprint,
        "version": str(version),
        "stage": mv.current_stage,
        "run_id": run_id,
        "metrics": client.get_run(run_id).data.metrics,
    }
