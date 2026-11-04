"""FastAPI: GET /recommend/{user_id}?k=10 -> HNSW retrieval of 100 candidates -> ranker -> top-k."""
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from .pipeline import build

BUDGET_MS = 50  # p95 target for one request (retrieval + ranking), measured in tests and run.py
state = {}


def load_system():
    # shortcut: trains at startup (~15 s); persist embeddings + index + ranker when this goes to production
    return build(epochs=int(os.environ.get("RECSYS_EPOCHS", "15")))[0]


@asynccontextmanager
async def lifespan(app):
    state["rec"] = load_system()
    yield
    state.clear()


app = FastAPI(title="two-tower-recsys", lifespan=lifespan)


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    rec = state["rec"]
    if not 0 <= user_id < len(rec.user_emb):
        raise HTTPException(404, "unknown user")
    t = time.perf_counter()
    items = rec.recommend(user_id, k)
    return {"user_id": user_id, "items": [int(i) for i in items], "latency_ms": (time.perf_counter() - t) * 1000}
