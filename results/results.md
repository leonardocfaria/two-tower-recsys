69172 train positives, 943 test users (leave-last-out), towers trained in 10.3 s.

| system | recall@10 | recall@50 | recall@100 | NDCG@10 |
|---|---|---|---|---|
| popularity | 0.093 | 0.218 | 0.306 | 0.064 |
| two-tower exact | 0.137 | 0.301 | 0.401 | 0.088 |
| two-tower HNSW | 0.137 | 0.301 | 0.402 | 0.088 |
| two-tower HNSW + ranker | 0.146 | 0.327 | 0.402 | 0.081 |

HNSW recall vs exact search @100: 0.992

API GET /recommend (HNSW 100 candidates + ranker, k=10), 500 requests in-process: p50 9.63 ms, p95 11.97 ms, p99 14.87 ms (budget p95 < 50 ms).
