69172 train positives, 943 test users (leave-last-out), towers trained in 10.0 s.

| system | recall@10 | recall@50 | recall@100 | NDCG@10 |
|---|---|---|---|---|
| popularity | 0.095 | 0.223 | 0.310 | 0.066 |
| two-tower exact | 0.139 | 0.305 | 0.406 | 0.092 |
| two-tower HNSW | 0.139 | 0.304 | 0.403 | 0.092 |
| two-tower HNSW + ranker | 0.153 | 0.323 | 0.403 | 0.092 |

HNSW recall vs exact search @100: 0.989

API GET /recommend (HNSW 100 candidates + ranker, k=10), 500 requests in-process: p50 6.07 ms, p95 8.38 ms, p99 9.70 ms (budget p95 < 50 ms).
