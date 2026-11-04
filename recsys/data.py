"""MovieLens-100k-shaped interactions: real files if present, else a seeded synthetic generator."""
from pathlib import Path

import numpy as np
import pandas as pd

N_USERS, N_ITEMS, N_GENRES = 943, 1682, 19
RAW = Path("data/raw/ml-100k")


def synthetic(n_users=N_USERS, n_items=N_ITEMS, n_ratings=100_000, seed=0):
    """Ratings from latent genre tastes + power-law popularity + noise. Returns (ratings, users, items)."""
    rng = np.random.default_rng(seed)
    genres = (rng.random((n_items, N_GENRES)) < 0.12).astype(np.float32)
    genres[np.arange(n_items), rng.integers(0, N_GENRES, n_items)] = 1  # every item has >= 1 genre
    taste = rng.dirichlet(np.full(N_GENRES, 0.3), n_users)  # each user likes a few genres
    pop = rng.pareto(2.0, n_items) + 1  # long tail, but not so extreme that popularity explains everything
    users = pd.DataFrame({"user": range(n_users), "age": rng.integers(0, 7, n_users),
                          "gender": rng.integers(0, 2, n_users), "occupation": rng.integers(0, 21, n_users)})
    # like ML-100k: every user rates >= 20 items, activity is long-tailed, ~n_ratings in total
    extra = rng.pareto(1.5, n_users)
    n_per_user = np.minimum(20 + extra / extra.sum() * (n_ratings - 20 * n_users), n_items // 2).astype(int)
    affinity = taste @ genres.T  # users x items
    rows = []
    for user in range(n_users):  # each user picks distinct items by popularity x affinity
        p = pop * (affinity[user] + 0.01) ** 2  # taste matters as much as popularity
        for item in rng.choice(n_items, n_per_user[user], replace=False, p=p / p.sum()):
            score = 3 + 6 * affinity[user, item] + rng.normal(0, 0.8)
            rows.append((user, item, int(np.clip(round(score), 1, 5))))
    ratings = pd.DataFrame(rows, columns=["user", "item", "rating"])
    ratings = ratings.sample(frac=1, random_state=seed).reset_index(drop=True)  # interleave users in time
    ratings["ts"] = np.arange(len(ratings))
    items = pd.DataFrame(genres, columns=[f"g{i}" for i in range(N_GENRES)]).assign(item=range(n_items))
    return ratings, users, items


def load(seed=0):
    """Real ML-100k from data/raw/ml-100k if present (ids re-based to 0), else synthetic."""
    if not (RAW / "u.data").exists():
        return synthetic(seed=seed)
    r = pd.read_csv(RAW / "u.data", sep="\t", names=["user", "item", "rating", "ts"])
    r[["user", "item"]] -= 1
    it = pd.read_csv(RAW / "u.item", sep="|", header=None, encoding="latin-1")
    items = it.iloc[:, 5:24].set_axis([f"g{i}" for i in range(N_GENRES)], axis=1).astype(np.float32).assign(item=it[0] - 1)
    us = pd.read_csv(RAW / "u.user", sep="|", names=["user", "age", "gender", "occupation", "zip"])
    users = pd.DataFrame({"user": us.user - 1, "age": np.minimum(us.age // 10, 6), "gender": (us.gender == "F").astype(int),
                          "occupation": us.occupation.astype("category").cat.codes})
    return r, users, items


def split(ratings: pd.DataFrame, min_pos: int = 3):
    """Implicit positives (rating >= 4). Per user, by time: last -> test, second last -> val, rest -> train.
    Users with fewer than min_pos positives stay entirely in train."""
    pos = ratings[ratings.rating >= 4].sort_values(["user", "ts"])
    rank = pos.groupby("user").cumcount(ascending=False)  # 0 = latest
    n = pos.groupby("user").user.transform("size")
    held = n >= min_pos
    test = pos[held & (rank == 0)]
    val = pos[held & (rank == 1)]
    train = pos[~held | (rank >= 2)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
