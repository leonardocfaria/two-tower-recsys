# two-tower-recsys

A two-stage recommender:
1. A **two-tower** retrieval model, searched with a **FAISS HNSW** index.
2. A lightweight **gradient-boosted ranker** that reorders the candidates.

It is evaluated with recall@k and NDCG@10 and served by FastAPI with a measured latency budget.

## Setup
```bash
cd ml/two-tower-recsys
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage
```bash
python -m pytest -q                 # 5 tests, ~10 s
python run.py                       # train + evaluate + API latency, ~25 s → results/results.{md,json}
uvicorn recsys.api:app --port 8000  # trains at startup (~15 s), then:
curl 'localhost:8000/recommend/42?k=10'
```

## Architecture
```
ratings ─► implicit positives (rating ≥ 4) ─► leave-last-out split per user: train │ val (2nd last) │ test (last)
                                   │
   user tower: id emb + age/gender/occupation emb + genre profile of history ─► MLP ─► 32-d, L2-normalised
   item tower: id emb + genre multi-hot ─► MLP ─► 32-d, L2-normalised
   loss: in-batch sampled softmax (τ = 0.05) with logQ correction (popular items are over-sampled as negatives)
                                   │
   serving score = u·v/τ + log q(item)   (logQ-trained towers score affinity; the prior adds popularity back)
   FAISS IndexHNSWFlat(inner product) over item vectors [v, log q, pad] ─► 100 unseen candidates
                                   │
   ranker: HistGradientBoosting on (tower score, retrieval rank, item popularity, user-genre affinity, user activity),
           trained on validation positives among their candidates ─► top-k
                                   │
   FastAPI GET /recommend/{user_id}?k=10
```
Details worth knowing:
- **Item padding for HNSW.** The log q column makes item norms unequal, and HNSW is unreliable for maximum inner product then: ANN recall dropped to 0.43 in a test. Each item therefore gets an extra √(M² − ‖v‖²) coordinate (and each user a 0). All items then have the same norm while every inner product stays the same, and ANN recall returns to 0.99.
- **Excluded items.** Everything the user has rated, whatever the rating, is never recommended. The only exception is the held-out item being predicted.
- **Leakage.** Per user, features come only from history before the evaluated item. A test checks that val and test interactions are later than all of the user's training interactions. Standard leave-last-out caveat: item popularity counts use all users' training data, part of which happened globally after a given user's test interaction.

**Data.** files.grouplens.org is blocked from this sandbox, so a seeded synthetic generator produces data in the exact MovieLens-100k shape:
- 943 users and 1682 items with 19 genres; user age bucket, gender and occupation;
- ratings from 1 to 5, at least 20 per user, about 79 k in total.

Item choice is ∝ popularity (Pareto tail) × (genre affinity)², and the rating depends on affinity. If `data/raw/ml-100k/u.data`, `u.item` and `u.user` are present, the real dataset is used instead.

## Results (seed 0, synthetic ML-100k shape, 4 vCPUs)
| system | recall@10 | recall@50 | recall@100 | NDCG@10 |
|---|---|---|---|---|
| popularity (unseen) | 0.095 | 0.223 | 0.310 | 0.066 |
| two-tower, exact search | 0.139 | 0.305 | **0.406** | **0.092** |
| two-tower, HNSW | 0.139 | 0.304 | 0.403 | **0.092** |
| two-tower HNSW + ranker | **0.153** | **0.323** | 0.403 | **0.092** |

- HNSW recall vs exact top-100: **0.989**. The ANN costs no measurable quality.
- Towers train in 10 s on CPU.
- **Serving:** 500 in-process requests (retrieval of 100 + ranking + k=10) give p50 6.1 ms, **p95 8.4 ms**, p99 9.7 ms against a **p95 < 50 ms budget**. The test suite enforces the budget.

Findings:
- **The two-tower model beats popularity** by 46 % on recall@10 and 39 % on NDCG@10. Its ceiling is the generator's true choice probability, which reached recall@50 0.376 during development, against the towers' ~0.30.
- **The ranker lifts recall@10 by 10 % (0.139 → 0.153) at equal NDCG@10.** It learns from only 943 positives, one validation item per user. More labelled interactions per user, or pairwise/listwise ranking, are the obvious next steps.
- **Self-review fix.** Excluding *every* rated item, not only positives, removed already-watched movies from the lists and raised all systems slightly. The first version let low-rated (but watched) movies be recommended. In that version the ranker even lowered NDCG@10 (0.088 → 0.081).

**Debugging notes.**
- **The data.** The first generator made the two-tower model lose to popularity. A ceiling check settled it: even the true choice probability reached recall@50 of only 0.50 against popularity's 0.42, so the synthetic data was popularity-dominated (max/median item popularity ≈ 10⁴). Sampling also collapsed onto a few heavy users. The fix was a milder popularity tail, a squared taste term and per-user sampling without replacement (≥ 20 ratings each, as in ML-100k), which opened a realistic gap: popularity 0.22, oracle 0.38.
- **logQ correction.** Removing it was much worse (recall@50 0.095 vs 0.365 at that point), so the correction stays. Adding the log q prior back at serving time is what improved NDCG.
- **Epochs.** More epochs eventually overfit the ID embeddings, so training uses 15 epochs on the final data.

## Not done (time box)
Persisted artifacts (the API trains at startup: `shortcut:` noted in `api.py`), cold-start users, hard-negative mining, a listwise ranker, hyper-parameter search on validation.
