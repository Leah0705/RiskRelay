# Load test

Measures the store-backed `/score` latency under concurrency. Two runners hit
the same endpoint:

- **`loadtest.py`** (asyncio + httpx) is scripted and repeatable. It warms the
  store, drives a fixed number of requests, and writes `results.md` with
  throughput and p50/p95/p99. It produced [results.md](results.md).
- **`locustfile.py`** opens the Locust UI for ramping users by hand.

## Run it

Start the API (Redis-backed via docker-compose, or in-memory locally):

```bash
# in-memory, single worker (matches results.md)
FEATURE_STORE_BACKEND=memory uvicorn src.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Then, in another shell:

```bash
python loadtest/loadtest.py \
    --host http://127.0.0.1:8000 \
    --accounts 2000 --warmup-per-account 5 \
    --requests 20000 --concurrency 8 \
    --hardware "your machine" --backend "in-memory" --server "uvicorn 1 worker"
```

Or interactively:

```bash
locust -f loadtest/locustfile.py --host http://127.0.0.1:8000
# open http://localhost:8089
```

## Notes

- The load test **warms per-account history first**, so the velocity windows
  hold real content. An empty store would make every lookup trivially cheap.
- Each `/score` request runs score-then-commit (`ingest: true`), so state grows
  during the run the same way it would in production.
- Throughput is bounded by a single CPU-bound Python worker; the scaling section
  in [results.md](results.md) covers the multi-worker + Redis path.
