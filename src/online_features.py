"""
online_features.py — real-time, stateful feature store for serving.

The batch pipeline in feature_engineering.py computes velocity / recency /
device features by scanning the WHOLE dataset. That's correct for training but
cannot run live at 1M+ events/day: at decision time you have one event
and milliseconds to score it, not the account's full history in a DataFrame.

This module maintains that history as incrementally-updated per-account state
and serves the same features the batch pipeline produces — with strict
train/serve parity (proven by tests/test_streaming_parity.py).

Two-phase, causal update contract
---------------------------------
    feats = store.get_features(event)   # state AS OF NOW, excluding this event
    ...score...
    store.commit(event)                 # fold this event in, for FUTURE events

get_features never sees the current event in its own state, exactly
mirroring the batch semantics (velocity excludes the current row; is_new_device
is evaluated against prior devices only). commit is what makes an event
count for the *next* one. Doing it in this order is what preserves causality —
the same guarantee the offline pipeline gets from sorting + shifting.

Backends
--------
- RedisFeatureStore: production path. Sliding-window velocity via sorted sets
  (ZADD/ZRANGEBYSCORE/ZREMRANGEBYSCORE), last-event string, device set. Commits
  are atomic per account via a MULTI pipeline. TTLs keep memory bounded.
- InMemoryFeatureStore: pure-Python fallback with identical behaviour and a
  per-account lock, so the repo runs and tests pass with no Redis server.

Select the backend with FEATURE_STORE_BACKEND=redis|memory (see get_feature_store).
"""

import os
import sys
import threading
from collections import defaultdict
from typing import Dict, Optional

import joblib
import pandas as pd

sys.path.append(".")
from src.config import FEATURES, PATHS
from src.feature_engineering import FIRST_EVENT_SENTINEL, velocity_window_seconds


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_epoch(ts) -> float:
    """Normalise any accepted timestamp form (epoch float, ISO string,
    pd.Timestamp, datetime) to epoch seconds — the score space used for the
    velocity sorted sets. Using one converter everywhere keeps batch and
    streaming on the same clock.
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    return pd.Timestamp(ts).timestamp()


class CounterpartyRisk:
    """Static counterparty-risk lookup, identical to apply_counterparty_risk_map /
    the API's _build_feature_row: smoothed target-encoded positive rate per
    counterparty (and per category) fit offline on the training split, with a
    global-mean fallback for unseen entities. This is NOT streaming state —
    it's a fixed map refreshed on retrain — so it's just a fast dict lookup.
    """

    def __init__(self, risk_maps: dict):
        self._counterparty = risk_maps["counterparty_risk"]
        self._category = risk_maps["category_risk"]
        self._global_mean = risk_maps["global_mean"]

    @classmethod
    def load(cls, path: str = None) -> "CounterpartyRisk":
        return cls(joblib.load(path or PATHS.counterparty_risk_map_path))

    def scores(self, counterparty_id, counterparty_category) -> Dict[str, float]:
        return {
            "counterparty_risk_score": self._counterparty.get(counterparty_id, self._global_mean),
            "category_risk_score": self._category.get(
                counterparty_category, self._global_mean
            ),
        }


class BaseFeatureStore:
    """Common feature assembly. Backends implement only the raw state
    primitives (_velocity_state, _last_event_epoch, _device_state, and their
    commit counterparts); this base turns them into the model's feature row so
    the two backends can never disagree on feature *definitions*.
    """

    def __init__(self, counterparty_risk: CounterpartyRisk):
        self.counterparty_risk = counterparty_risk
        self.window_seconds = velocity_window_seconds()

    # ---- feature assembly (shared) ----

    def get_features(self, event: dict) -> Dict[str, float]:
        """Full stateful feature vector for `event`, computed from state as of now
        and EXCLUDING event itself. PCA (V1-V28) and Amount are supplied by the
        caller/request — they aren't stateful — so they're not returned here.
        """
        acct = event[FEATURES.account_col]
        now = _to_epoch(event["timestamp"])
        device = event[FEATURES.device_col]

        feats: Dict[str, float] = {}

        # velocity: count + summed amount over each trailing window, priors only
        vel = self._velocity_state(acct, now)
        for w in FEATURES.velocity_windows_minutes:
            count, amount_sum = vel[w]
            feats[f"velocity_count_{w}m"] = float(count)
            feats[f"velocity_amount_{w}m"] = float(max(amount_sum, 0.0))

        # recency: sentinel on first-ever event, never 0
        last = self._last_event_epoch(acct)
        feats["seconds_since_last_event"] = (
            FIRST_EVENT_SENTINEL if last is None else float(now - last)
        )

        # device consistency: evaluate is_new BEFORE this device is committed
        seen, distinct_prior = self._device_state(acct, device)
        feats["is_new_device"] = int(not seen)
        feats["distinct_device_count_so_far"] = int(
            distinct_prior + (0 if seen else 1)
        )

        feats.update(
            self.counterparty_risk.scores(
                event[FEATURES.counterparty_col], event[FEATURES.counterparty_category_col]
            )
        )
        return feats

    def commit(self, event: dict) -> None:
        """Fold `event` into per-account state so it counts for future events."""
        raise NotImplementedError

    # ---- state primitives (backend-specific) ----

    def _velocity_state(self, acct, now) -> Dict[int, tuple]:
        raise NotImplementedError

    def _last_event_epoch(self, acct) -> Optional[float]:
        raise NotImplementedError

    def _device_state(self, acct, device) -> tuple:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory backend (no Redis required)
# ---------------------------------------------------------------------------

class InMemoryFeatureStore(BaseFeatureStore):
    """Process-local state. Behaviourally identical to the Redis backend; used
    for tests and for running the repo without a Redis server. A per-account
    lock makes commits atomic under concurrent workers (mirrors Redis MULTI).
    """

    def __init__(self, counterparty_risk: CounterpartyRisk):
        super().__init__(counterparty_risk)
        self._events = defaultdict(list)      # acct -> list[(epoch, amount)]
        self._last = {}                        # acct -> epoch
        self._devices = defaultdict(set)       # acct -> {device_id}
        self._locks = defaultdict(threading.Lock)
        self._max_window = max(self.window_seconds.values())

    def _velocity_state(self, acct, now):
        events = self._events.get(acct, [])
        out = {}
        for w in FEATURES.velocity_windows_minutes:
            lower = now - self.window_seconds[w]
            # left-exclusive, right-inclusive to match pandas rolling(closed="right")
            in_window = [a for (ts, a) in events if lower < ts <= now]
            out[w] = (len(in_window), sum(in_window))
        return out

    def _last_event_epoch(self, acct):
        return self._last.get(acct)

    def _device_state(self, acct, device):
        devs = self._devices.get(acct, set())
        return (device in devs, len(devs))

    def commit(self, event: dict) -> None:
        acct = event[FEATURES.account_col]
        now = _to_epoch(event["timestamp"])
        amount = float(event[FEATURES.amount_col])
        with self._locks[acct]:
            events = self._events[acct]
            events.append((now, amount))
            # trim anything older than the largest window keeps memory bounded
            cutoff = now - self._max_window
            if events and events[0][0] <= cutoff:
                self._events[acct] = [(ts, a) for (ts, a) in events if ts > cutoff]
            self._last[acct] = now
            self._devices[acct].add(event[FEATURES.device_col])


# ---------------------------------------------------------------------------
# Redis backend (production path)
# ---------------------------------------------------------------------------

class RedisFeatureStore(BaseFeatureStore):
    """Redis-backed sliding-window aggregator.

    Key schema (per account_id):
      fs:vel:{acct}   sorted set  score = event epoch seconds
                                  member = "{epoch}|{amount}|{event_id}"
                      -> ZRANGEBYSCORE (now-w, now] gives the window's members;
                         count = len, amount = sum of the parsed amount field.
                         The event_id keeps members unique when an account has
                         two events at the same epoch+amount.
      fs:last:{acct}  string      last event epoch (seconds_since_last_event source)
      fs:dev:{acct}   set         distinct device_ids seen so far

    Why sorted sets: a time-scored ZSET is the natural sliding window — old
    events age out with one ZREMRANGEBYSCORE and any window is a single
    range query, all O(log N + K). TTLs bound memory for dormant accounts.
    """

    # velocity events only need to live as long as the largest window; last-event
    # and device history are longer-lived signals, kept for a rolling horizon.
    DEVICE_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days

    def __init__(self, counterparty_risk: CounterpartyRisk, client):
        super().__init__(counterparty_risk)
        self.r = client
        self._max_window = max(self.window_seconds.values())

    @staticmethod
    def _k_vel(acct): return f"fs:vel:{acct}"

    @staticmethod
    def _k_last(acct): return f"fs:last:{acct}"

    @staticmethod
    def _k_dev(acct): return f"fs:dev:{acct}"

    def _velocity_state(self, acct, now):
        key = self._k_vel(acct)
        out = {}
        # one range read per window; each is left-exclusive, right-inclusive
        pipe = self.r.pipeline(transaction=False)
        windows = FEATURES.velocity_windows_minutes
        for w in windows:
            lower = now - self.window_seconds[w]
            pipe.zrangebyscore(key, f"({lower}", now)
        results = pipe.execute()
        for w, members in zip(windows, results):
            total = 0.0
            for m in members:
                m = m.decode() if isinstance(m, bytes) else m
                # member = "epoch|amount|event_id"
                total += float(m.split("|", 2)[1])
            out[w] = (len(members), total)
        return out

    def _last_event_epoch(self, acct):
        v = self.r.get(self._k_last(acct))
        if v is None:
            return None
        return float(v.decode() if isinstance(v, bytes) else v)

    def _device_state(self, acct, device):
        key = self._k_dev(acct)
        pipe = self.r.pipeline(transaction=False)
        pipe.sismember(key, device)
        pipe.scard(key)
        seen, count = pipe.execute()
        return (bool(seen), int(count))

    def commit(self, event: dict) -> None:
        acct = event[FEATURES.account_col]
        now = _to_epoch(event["timestamp"])
        amount = float(event[FEATURES.amount_col])
        event_id = event.get("event_id", "")
        vel_key = self._k_vel(acct)
        member = f"{now}|{amount}|{event_id}"

        # MULTI makes the whole commit atomic for this account, so concurrent
        # workers can't interleave a partial update.
        pipe = self.r.pipeline(transaction=True)
        pipe.zadd(vel_key, {member: now})
        pipe.zremrangebyscore(vel_key, "-inf", now - self._max_window)
        pipe.expire(vel_key, int(self._max_window * 2))
        pipe.set(self._k_last(acct), now, ex=self.DEVICE_TTL_SECONDS)
        pipe.sadd(self._k_dev(acct), event[FEATURES.device_col])
        pipe.expire(self._k_dev(acct), self.DEVICE_TTL_SECONDS)
        pipe.execute()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_feature_store(
    backend: str = None,
    counterparty_risk: CounterpartyRisk = None,
    redis_url: str = None,
) -> BaseFeatureStore:
    """Build the feature store selected by config/env.

    FEATURE_STORE_BACKEND = "redis" | "memory" (default "memory" so the repo
    runs with zero infra). REDIS_URL configures the Redis connection.
    """
    backend = (backend or os.getenv("FEATURE_STORE_BACKEND", "memory")).lower()
    counterparty_risk = counterparty_risk or CounterpartyRisk.load()

    if backend == "redis":
        import redis
        url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.Redis.from_url(url)
        client.ping()  # fail fast if Redis is unreachable
        return RedisFeatureStore(counterparty_risk, client)

    if backend == "memory":
        return InMemoryFeatureStore(counterparty_risk)

    raise ValueError(f"Unknown FEATURE_STORE_BACKEND: {backend!r} (use 'redis' or 'memory')")
