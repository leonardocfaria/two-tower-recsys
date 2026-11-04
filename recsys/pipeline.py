"""Retrieval (FAISS HNSW) + ranking (HistGradientBoosting) + offline metrics."""
import math
import time

import faiss
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import load, split
from .model import train_towers

N_CANDIDATES = 100


def recall_at_k(recs, gold, k) -> float:
    return float(gold in list(recs[:k]))


def ndcg_at_k(recs, gold, k) -> float:
    """One relevant item per user, so IDCG = 1 and NDCG = 1 / log2(rank + 2)."""
    recs = list(recs[:k])
    return 1 / math.log2(recs.index(gold) + 2) if gold in recs else 0.0


class Recommender:
    def __init__(self, user_emb, item_emb, history, items, temp=0.05, ef_search=128, exclude=None):
        """history: positives used for features and the popularity prior. exclude: every (user, item) already
        rated, whatever the rating (default: history) - never recommended.
        The towers were trained with logQ correction, so u·v/temp scores affinity without popularity. Adding
        log q(item) back gives the full estimate of log p(item | user). Augmenting the vectors as
        user [u/temp, 1] · item [v, log q] keeps that a single inner-product ANN search."""
        n_users, n_items = len(user_emb), len(item_emb)
        counts = np.bincount(history.item, minlength=n_items)
        log_q = np.log((counts + 1) / (counts.sum() + n_items)).astype(np.float32)
        self.user_emb = np.hstack([user_emb / temp, np.ones((n_users, 1), np.float32)]).astype(np.float32)
        self.item_emb = np.hstack([item_emb, log_q[:, None]]).astype(np.float32)
        # HNSW handles inner product poorly when item norms differ (the log q column makes them differ).
        # MIPS -> equal-norm trick: items get sqrt(M^2 - |v|^2) as an extra dim, users get 0: same scores.
        norms = np.linalg.norm(self.item_emb, axis=1)
        pad = np.sqrt(norms.max() ** 2 - norms ** 2)[:, None]
        self.index = faiss.IndexHNSWFlat(self.item_emb.shape[1] + 1, 32, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.add(np.hstack([self.item_emb, pad]).astype(np.float32))
        self.seen = [set() for _ in range(n_users)]
        exclude = history if exclude is None else exclude
        for u, i in zip(exclude.user, exclude.item):
            self.seen[u].add(i)
        self.popular = np.argsort(-counts, kind="stable")
        self.log_pop = np.log1p(counts)
        G = items.sort_values("item").filter(like="g").to_numpy()
        self.genres = G / np.maximum(G.sum(1, keepdims=True), 1)
        prof = np.zeros((n_users, G.shape[1]))
        np.add.at(prof, history.user.to_numpy(), G[history.item.to_numpy()])
        self.profile = prof / np.maximum(prof.sum(1, keepdims=True), 1)
        self.activity = np.log1p(np.bincount(history.user, minlength=n_users))
        self.ranker = None

    def retrieve(self, u, n=N_CANDIDATES, exact=False):
        """Top-n unseen items by tower score. exact=True uses brute force instead of HNSW."""
        q = self.user_emb[u: u + 1]
        m = n + len(self.seen[u])
        if exact:
            s = (self.item_emb @ q[0])
            ids = np.argsort(-s, kind="stable")[:m]
            scores = s[ids]
        else:
            scores, ids = self.index.search(np.hstack([q, np.zeros((1, 1), np.float32)]), m)
            scores, ids = scores[0], ids[0]
        keep = [k for k, i in enumerate(ids) if i >= 0 and i not in self.seen[u]][:n]
        return ids[keep], scores[keep]

    def features(self, u, ids, scores):
        return np.column_stack([scores, np.arange(len(ids)), self.log_pop[ids],
                                self.genres[ids] @ self.profile[u], np.full(len(ids), self.activity[u])])

    def fit_ranker(self, positives, seed=0):
        """Learn to rank the held-out positive among each user's retrieved candidates."""
        X, y = [], []
        for u, gold in zip(positives.user, positives.item):
            ids, scores = self.retrieve(u)
            X.append(self.features(u, ids, scores))
            y.append(ids == gold)
        self.ranker = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=seed)
        self.ranker.fit(np.vstack(X), np.concatenate(y))
        return self

    def recommend(self, u, k=10, rank=True, exact=False):
        ids, scores = self.retrieve(u, exact=exact)
        if rank and self.ranker is not None:
            order = np.argsort(-self.ranker.predict_proba(self.features(u, ids, scores))[:, 1], kind="stable")
            ids = ids[order]
        return ids[:k]

    def popular_unseen(self, u, k=10):
        return [i for i in self.popular[: k + len(self.seen[u])] if i not in self.seen[u]][:k]


def build(epochs=15, seed=0, data=None):
    """Train towers on train, fit the ranker on validation, return a recommender for serving/testing
    (history = train + val, so the val item counts as seen) plus the held-out test positives."""
    ratings, users, items = data or load(seed)
    train, val, test = split(ratings)
    t = time.perf_counter()
    _, ue, ie = train_towers(train, users, items, epochs=epochs, seed=seed)
    train_s = time.perf_counter() - t
    # everything the user rated (any rating) is excluded, except the held-out item(s) being predicted
    held_val = ratings.merge(pd.concat([val, test])[["user", "item"]], how="left", indicator=True)
    held_test = ratings.merge(test[["user", "item"]], how="left", indicator=True)
    ranker = Recommender(ue, ie, train, items, exclude=held_val[held_val._merge == "left_only"]).fit_ranker(val, seed).ranker
    rec = Recommender(ue, ie, pd.concat([train, val]), items, exclude=held_test[held_test._merge == "left_only"])
    rec.ranker = ranker
    return rec, test, {"train_seconds": train_s, "n_train": len(train), "n_test": len(test)}


def evaluate(rec, test, ks=(10, 50, 100)) -> dict:
    systems = {
        "popularity": lambda u: rec.popular_unseen(u, max(ks)),
        "two-tower exact": lambda u: rec.recommend(u, max(ks), rank=False, exact=True),
        "two-tower HNSW": lambda u: rec.recommend(u, max(ks), rank=False),
        "two-tower HNSW + ranker": lambda u: rec.recommend(u, max(ks), rank=True),
    }
    out = {}
    for name, fn in systems.items():
        recs = [(fn(u), g) for u, g in zip(test.user, test.item)]
        row = {f"recall@{k}": float(np.mean([recall_at_k(r, g, k) for r, g in recs])) for k in ks}
        row["ndcg@10"] = float(np.mean([ndcg_at_k(r, g, 10) for r, g in recs]))
        out[name] = row
    overlap = [len(set(rec.retrieve(u)[0]) & set(rec.retrieve(u, exact=True)[0])) / N_CANDIDATES for u in test.user]
    out["ann_recall_vs_exact@100"] = float(np.mean(overlap))
    return out
