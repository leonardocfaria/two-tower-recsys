# two-tower-recsys

## Goal
Build "Two-tower recommender with ANN retrieval and ranking". Train a two-tower retrieval model on MovieLens-100k, build an ANN index (FAISS or hnswlib), add a lightweight ranking stage, evaluate recall@k/NDCG, and serve recommendations from an API with a latency budget. Keep it CPU-friendly (the cloud box has no GPU): small models and datasets, a fixed random seed, and a documented runtime.

## Definition of done
- [x] retrieval and ranking trained
- [x] recall@k and NDCG reported
- [x] API serves top-k under a measured latency
- [x] Tests pass with one command; CI workflow runs them
- [x] README with setup, usage, architecture and results
- [x] Self code review done
- [ ] Draft PR open

