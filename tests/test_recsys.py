import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from recsys.data import synthetic, split
from recsys.pipeline import build, evaluate, ndcg_at_k, recall_at_k

SMALL = dict(n_users=300, n_items=500, n_ratings=20_000, seed=0)


def test_metrics():
    assert recall_at_k([5, 3, 9], 3, 1) == 0 and recall_at_k([5, 3, 9], 3, 2) == 1
    assert ndcg_at_k([3, 5], 3, 10) == 1.0
    assert ndcg_at_k([5, 3], 3, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k([5, 3], 7, 10) == 0


def test_synthetic_shape_and_split_has_no_time_leakage():
    r, users, items = synthetic(**SMALL)
    assert r.equals(synthetic(**SMALL)[0])
    assert r.rating.between(1, 5).all() and not r.duplicated(["user", "item"]).any()
    train, val, test = split(r)
    assert test.user.is_unique and val.user.is_unique
    last_train = train.groupby("user").ts.max()
    for name, df in (("val", val), ("test", test)):
        assert (df.set_index("user").ts > last_train.reindex(df.user).to_numpy()).all(), name
    assert (test.set_index("user").ts > val.set_index("user").ts.reindex(test.user)).all()


@pytest.fixture(scope="module")
def system():
    return build(epochs=8, data=synthetic(**SMALL))


def test_two_tower_beats_popularity_and_hnsw_matches_exact(system):
    rec, test, _ = system
    m = evaluate(rec, test)
    assert m["two-tower exact"]["recall@50"] > m["popularity"]["recall@50"]
    assert m["ann_recall_vs_exact@100"] > 0.95
    for row in (v for k, v in m.items() if isinstance(v, dict)):
        assert row["recall@10"] <= row["recall@50"] <= row["recall@100"]


def test_recommendations_exclude_seen_items(system):
    rec, test, _ = system
    for u in test.user[:50]:
        out = rec.recommend(u, 20)
        assert len(out) == 20 and len(set(out)) == 20 and not set(out) & rec.seen[u]
    r, *_ = synthetic(**SMALL)
    low = r[r.rating < 4]
    u, i = low.user.iloc[0], low.item.iloc[0]
    assert i in rec.seen[u]  # a movie rated low is watched too: never recommended


def test_api_serves_topk_within_budget(system, monkeypatch):
    import recsys.api as api

    monkeypatch.setattr(api, "load_system", lambda: system[0])
    with TestClient(api.app) as client:
        assert client.get("/recommend/99999").status_code == 404
        r = client.get("/recommend/3?k=5").json()
        assert len(r["items"]) == 5
        lat = []
        for u in range(100):
            t = time.perf_counter()
            assert client.get(f"/recommend/{u}?k=10").status_code == 200
            lat.append(time.perf_counter() - t)
    assert np.percentile(lat, 95) * 1000 < api.BUDGET_MS
