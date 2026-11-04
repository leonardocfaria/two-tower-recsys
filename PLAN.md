# two-tower-recsys Implementation Plan

## Brainstorm (answered from SPEC + repo evidence)
- **What is it?** The standard two-stage recommender: a two-tower retrieval model with an ANN index for fast candidate generation, then a light ranker that reorders candidates. It is evaluated offline with recall@k and NDCG and served with a measured latency.
- **Data:** files.grouplens.org is blocked (DECISIONS #3). The synthetic generator has the exact MovieLens-100k shape: 943 users, 1682 items, ~100 k ratings from 1-5, 19 genres, user age/gender/occupation. Ratings come from latent user tastes over genres, plus item popularity on a power law, plus noise. Timestamps are increasing. `data/raw/ml-100k/u.data` (and `u.item`) are used instead if present.
- **Task:** implicit feedback, rating ≥ 4 is a positive. Split leave-last-out per user by time: last positive → test, second last → validation (used to train the ranker), the rest → train. Seen train items are excluded from recommendations.
- **Retrieval:** user tower (id embedding + age/gender/occupation embeddings → MLP) and item tower (id embedding + genre multi-hot → MLP), 32-d L2-normalised. Trained with in-batch sampled softmax (temperature 0.05) with logQ popularity correction. FAISS `IndexHNSWFlat` with inner product; exact search is kept for the ANN-recall check.
- **Ranking:** sklearn HistGradientBoosting over (user, candidate) features: tower score, retrieval rank, item popularity, user-genre affinity · item genres, user activity. Trained on validation positives among top-100 candidates, evaluated on test.
- **Serving:** FastAPI `GET /recommend/{user_id}?k=10` → retrieve 100 via HNSW → rank → top-k. In-process p50/p95 over 500 requests, budget p95 < 50 ms.
- **Non-goals:** real-time feature updates, cold-start users, a multi-objective ranker.

## Files
- `recsys/data.py`, `recsys/model.py` (towers + train), `recsys/pipeline.py` (index, candidates, ranker, metrics, `build()` / `evaluate()`), `recsys/api.py`, `run.py`.

## Tasks (TDD)
- [x] 1. data shape/determinism; leave-last-out split has no time leakage; metrics (recall@k, NDCG) on toy lists.
- [x] 2. towers learn: recall@50 beats the popularity baseline on a small config; HNSW recall vs exact ≥ 0.95.
- [x] 3. ranker improves NDCG@10 over retrieval order (or at least runs); seen items excluded.
- [x] 4. API returns k unseen items under budget; run.py report; README; CI.
