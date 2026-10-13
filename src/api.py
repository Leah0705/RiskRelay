"""
api.py — FastAPI service exposing the trained XGBoost risk model for
real-time scoring. Designed for the shape of request an upstream system
sends at decision time, before it accepts an event.

Two ways to supply the time-dependent features (velocity, recency, device):

  1. Store-backed (production path): set "use_feature_store": true and the
     service fetches those features from the online feature store
     (src/online_features.py) using the event's ids — computed live from
     per-account streaming state, not passed by the caller. Only the raw
     event fields (V1-V28, Amount, ids, timestamp) are required.

  2. Caller-provided (legacy path, default): the caller passes all streaming
     features pre-computed in `features`. Kept for backward compatibility.

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

Example request (store-backed):
    POST /score
    {
      "event_id": "EVT_00012345",
      "account_id": "ACCT_004821",
      "counterparty_id": 118,
      "counterparty_category": "cat_1",
      "device_id": "DEV_002_ACCT_004821",
      "amount": 482.10,
      "timestamp": "2026-07-09T14:32:00Z",
      "use_feature_store": true,
      "ingest": true,
      "features": { "V1": -1.2, "V2": 0.5, ... }   # V1-V28 only in this mode
    }
"""

import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional

from src.config import FEATURES, MODEL, REGISTRY
from src.explainability import explain_single_event
from src.online_features import CounterpartyRisk, get_feature_store
from src import registry

# ---- artifacts + online store, loaded once at startup (see lifespan) ----
_model = None
_feature_cols = None
_threshold_info = None
_counterparty_risk_maps = None
_device_map = None
_explainer = None
_feature_store = None
_model_version = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CURRENT PRODUCTION model from the model registry, and construct
    the online feature store, once at process start.

    The model is resolved BY STAGE (`Production`) from the MLflow registry — not
    from a hardcoded joblib path — together with the exact threshold, feature
    list, and encoders that were registered alongside it. This guarantees the
    served threshold and feature order always match the served model.

    How a new Production promotion is picked up: at STARTUP. Promoting a new
    champion (src/retraining.py) and restarting the API (or its workers) is what
    ships it. A running process keeps serving the version it loaded — a
    deliberate, safe default: no mid-flight model swaps under load. A real
    deployment would trigger a rolling restart on promotion (or add a guarded
    hot-reload endpoint); noted here rather than hidden.
    """
    global _model, _feature_cols, _threshold_info, _counterparty_risk_maps
    global _device_map, _explainer, _feature_store, _model_version

    bundle = registry.load_bundle(stage=REGISTRY.production_stage)
    _model = bundle["model"]

    # Serving scores one row at a time, so pin inference to a single thread:
    # see MODEL.serving_n_jobs for why this differs from the training setting.
    # Wrapped because not every estimator that could be registered exposes it.
    if MODEL.serving_n_jobs:
        try:
            _model.set_params(n_jobs=MODEL.serving_n_jobs)
        except (ValueError, AttributeError, TypeError):
            print(f"note: {type(_model).__name__} takes no n_jobs; "
                  f"serving thread count left as trained.")
    _feature_cols = bundle["feature_cols"]
    _threshold_info = bundle["threshold_info"]
    _counterparty_risk_maps = bundle["counterparty_risk_maps"]
    _device_map = bundle["device_map"]
    _model_version = bundle["version"]

    # TreeExplainer is cheap relative to scoring volume; build once and cache.
    import shap
    _explainer = shap.TreeExplainer(_model)

    # Online feature store (Redis or in-memory, per FEATURE_STORE_BACKEND).
    # Reuse the already-loaded counterparty-risk map rather than loading it twice.
    _feature_store = get_feature_store(counterparty_risk=CounterpartyRisk(_counterparty_risk_maps))

    print(f"Loaded {REGISTRY.registered_model_name} v{_model_version} "
          f"(stage={REGISTRY.production_stage}) from the registry, plus feature "
          f"list, threshold, encoders, and feature store.")
    yield
    # nothing to tear down explicitly; Redis connections are pooled


app = FastAPI(title="RiskRelay Scoring API", version="1.1.0", lifespan=lifespan)


class EventRequest(BaseModel):
    event_id: str
    account_id: str
    counterparty_id: int
    counterparty_category: str
    device_id: str
    amount: float
    timestamp: str
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Model input features. In store-backed mode (use_feature_store=true) "
                    "supply only V1-V28 — the velocity/recency/device features come from "
                    "the online feature store. In legacy mode supply all of them "
                    "pre-computed (V1-V28, seconds_since_last_event, velocity_*_10m/60m/1440m, "
                    "is_new_device, distinct_device_count_so_far).",
    )
    use_feature_store: bool = Field(
        default=False,
        description="If true, fetch streaming features from the online store instead of "
                    "requiring the caller to pass them.",
    )
    ingest: bool = Field(
        default=False,
        description="Store-backed mode only: after scoring, commit this event into "
                    "the feature store so it counts toward future events' features. "
                    "Scoring always happens BEFORE the commit (causal ordering).",
    )
    explain: Optional[bool] = False


class IngestRequest(BaseModel):
    event_id: str
    account_id: str
    counterparty_id: int
    counterparty_category: str
    device_id: str
    amount: float
    timestamp: str


class ScoreResponse(BaseModel):
    event_id: str
    risk_probability: float
    is_flagged: bool
    threshold_used: float
    latency_ms: float
    top_factors: Optional[list] = None


def _event_dict(req) -> dict:
    """Normalise a request into the dict shape the feature store expects."""
    return {
        "event_id": req.event_id,
        FEATURES.account_col: req.account_id,
        FEATURES.counterparty_col: req.counterparty_id,
        FEATURES.counterparty_category_col: req.counterparty_category,
        FEATURES.device_col: req.device_id,
        FEATURES.amount_col: req.amount,
        "timestamp": req.timestamp,
    }


def _finalize_row(row: dict) -> pd.DataFrame:
    """Validate all model columns are present and order them for the model."""
    missing = [c for c in _feature_cols if c not in row]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}. "
                   f"Expected all of: {_feature_cols}",
        )
    return pd.DataFrame([{c: row[c] for c in _feature_cols}])


def _build_feature_row_legacy(req: EventRequest) -> pd.DataFrame:
    """Legacy path: caller passes all streaming features pre-computed."""
    row = dict(req.features)
    row["Amount"] = req.amount

    counterparty = _counterparty_risk_maps["counterparty_risk"].get(
        req.counterparty_id, _counterparty_risk_maps["global_mean"]
    )
    category = _counterparty_risk_maps["category_risk"].get(
        req.counterparty_category, _counterparty_risk_maps["global_mean"]
    )
    row.setdefault("counterparty_risk_score", counterparty)
    row.setdefault("category_risk_score", category)
    return _finalize_row(row)


def _build_feature_row_from_store(req: EventRequest) -> pd.DataFrame:
    """Store-backed path: stateful features come from the online store; the
    caller supplies only the raw event inputs (V1-V28) plus Amount.
    """
    row = dict(req.features)              # V1-V28
    row["Amount"] = req.amount
    row.update(_feature_store.get_features(_event_dict(req)))  # velocity/recency/device/counterparty
    return _finalize_row(row)


@app.post("/score", response_model=ScoreResponse)
def score_event(req: EventRequest):
    start = time.perf_counter()

    if req.use_feature_store:
        X = _build_feature_row_from_store(req)
    else:
        X = _build_feature_row_legacy(req)

    proba = float(_model.predict_proba(X)[:, 1][0])
    threshold = _threshold_info["best_threshold"]
    is_flagged = proba >= threshold

    top_factors = None
    if req.explain:
        top_factors = explain_single_event(_explainer, X)

    # Causal ordering: the event is scored against state that EXCLUDES it,
    # then (optionally) committed so it counts for future events.
    if req.use_feature_store and req.ingest:
        _feature_store.commit(_event_dict(req))

    latency_ms = (time.perf_counter() - start) * 1000

    return ScoreResponse(
        event_id=req.event_id,
        risk_probability=round(proba, 6),
        is_flagged=is_flagged,
        threshold_used=threshold,
        latency_ms=round(latency_ms, 2),
        top_factors=top_factors,
    )


@app.post("/ingest")
def ingest_event(req: IngestRequest):
    """Fold an event into the online feature store without scoring it.
    Used to warm per-account state (e.g. load test warmup, backfill) and as a
    pure ingestion path separate from scoring.
    """
    _feature_store.commit(_event_dict(req))
    return {"event_id": req.event_id, "ingested": True}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,      # which registry version is serving
        "model_stage": REGISTRY.production_stage,
    }
