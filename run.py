"""Train + evaluate + measure API latency. python run.py [--out results]   (~1 min on 4 vCPUs)"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

import recsys.api as api
from recsys.pipeline import build, evaluate


def main(out="results", epochs=15):
    rec, test, info = build(epochs=epochs)
    metrics = evaluate(rec, test)
    api.load_system = lambda: rec  # serve the model we just evaluated
    lat = []
    with TestClient(api.app) as client:
        for n in range(600):
            t = time.perf_counter()
            client.get(f"/recommend/{n % len(rec.user_emb)}?k=10").raise_for_status()
            if n >= 100:  # first 100 are warm-up
                lat.append((time.perf_counter() - t) * 1000)
    serving = {"requests": len(lat), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
               "p99_ms": float(np.percentile(lat, 99)), "budget_ms": api.BUDGET_MS}
    res = {**info, "metrics": metrics, "serving": serving}
    out = Path(out)
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    lines = [f"{info['n_train']} train positives, {info['n_test']} test users (leave-last-out), towers trained in "
             f"{info['train_seconds']:.1f} s.", "", "| system | recall@10 | recall@50 | recall@100 | NDCG@10 |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v['recall@10']:.3f} | {v['recall@50']:.3f} | {v['recall@100']:.3f} | {v['ndcg@10']:.3f} |"
              for k, v in metrics.items() if isinstance(v, dict)]
    lines += ["", f"HNSW recall vs exact search @100: {metrics['ann_recall_vs_exact@100']:.3f}", "",
              f"API GET /recommend (HNSW 100 candidates + ranker, k=10), {serving['requests']} requests in-process: "
              f"p50 {serving['p50_ms']:.2f} ms, p95 {serving['p95_ms']:.2f} ms, p99 {serving['p99_ms']:.2f} ms "
              f"(budget p95 < {api.BUDGET_MS} ms)."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    main(p.parse_args().out)
