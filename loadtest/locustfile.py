"""
locustfile.py — interactive alternative to loadtest.py, for exploring the
latency/throughput curve with Locust's live web UI.

The scripted loadtest.py is what produces loadtest/results.md (deterministic,
reports p50/p95/p99 in one shot). Use this when you want to ramp users
interactively and watch the distribution move.

Run:
    locust -f loadtest/locustfile.py --host http://localhost:8000
    # then open http://localhost:8089

Each simulated user warms one account's history on start, then repeatedly
scores store-backed events for that account (score-then-ingest).
"""

import random
import time
from datetime import datetime, timezone

from locust import HttpUser, task, between

CATEGORIES = ["cat_0", "cat_1", "cat_2", "cat_3", "cat_4", "cat_5"]


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _features():
    return {f"V{i}": random.gauss(0, 1) for i in range(1, 29)}


class RiskScoringUser(HttpUser):
    wait_time = between(0.0, 0.05)

    def on_start(self):
        # give this virtual user its own account with some recent history so
        # velocity lookups are non-empty from the first scored request
        self.account_id = f"ACCT_{random.randint(0, 100000):06d}"
        now = time.time()
        for _ in range(5):
            self.client.post("/ingest", json={
                "event_id": f"WARM_{self.account_id}_{random.random()}",
                "account_id": self.account_id,
                "counterparty_id": random.randint(1, 500),
                "counterparty_category": random.choice(CATEGORIES),
                "device_id": f"{self.account_id}_dev{random.randint(0,2)}",
                "amount": round(random.uniform(1, 800), 2),
                "timestamp": _iso(now - random.uniform(0, 24 * 3600)),
            })

    @task
    def score(self):
        self.client.post("/score", json={
            "event_id": f"LOAD_{random.random()}",
            "account_id": self.account_id,
            "counterparty_id": random.randint(1, 500),
            "counterparty_category": random.choice(CATEGORIES),
            "device_id": f"{self.account_id}_dev{random.randint(0,3)}",
            "amount": round(random.uniform(1, 800), 2),
            "timestamp": _iso(time.time()),
            "use_feature_store": True,
            "ingest": True,
            "features": _features(),
        })
