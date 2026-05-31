"""
scripts/benchmark.py
Latency benchmark for the TrialMate matching pipeline.

Loads the pipeline, runs N synthetic patient queries (default N=50), and
reports p50/p95/p99 latency in milliseconds.  Results are saved to
evaluation/benchmark_results.json.

Usage:
    python scripts/benchmark.py [--n 50] [--top-k 5]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Synthetic patient generator (no external deps needed)
# ---------------------------------------------------------------------------

CONDITIONS_POOL = [
    ["type 2 diabetes mellitus"],
    ["type 2 diabetes mellitus", "hypertension"],
    ["hypertension"],
    ["non-small cell lung cancer", "stage IV"],
    ["hypertension", "chronic kidney disease stage 3"],
    ["type 2 diabetes mellitus", "non-alcoholic fatty liver disease"],
    ["small cell lung cancer", "limited stage"],
    ["type 2 diabetes mellitus", "obesity"],
    ["lung adenocarcinoma", "stage IIIB"],
    ["hypertension", "heart failure with reduced ejection fraction"],
]
MEDS_POOL = [
    ["metformin"],
    ["metformin", "lisinopril"],
    ["amlodipine", "losartan"],
    ["carboplatin", "paclitaxel"],
    ["pembrolizumab"],
    ["insulin glargine"],
    ["semaglutide", "metformin"],
    [],
]


def _make_patient(i: int, rng: random.Random) -> dict:
    return {
        "patient_id": f"BENCH_{i:04d}",
        "age": rng.randint(25, 80),
        "sex": rng.choice(["male", "female"]),
        "conditions": rng.choice(CONDITIONS_POOL),
        "medications": rng.choice(MEDS_POOL),
        "labs": {
            "hemoglobin_a1c": round(rng.uniform(6.5, 11.0), 1),
            "creatinine": round(rng.uniform(0.6, 2.5), 2),
            "egfr": rng.randint(20, 100),
        },
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_benchmark(n: int = 50, top_k: int = 5) -> dict:
    from matching.scorer import MatchingPipeline
    from api.patient_schema import PatientInput

    print(f"Loading pipeline…")
    pipeline = MatchingPipeline.load()
    n_trials = len(pipeline.trials)
    print(f"Pipeline ready — {n_trials} trials indexed.")

    rng = random.Random(42)
    patients = [PatientInput(**_make_patient(i, rng)) for i in range(n)]

    latencies_ms: list[float] = []
    print(f"Running {n} patient queries (top_k={top_k})…")
    for idx, patient in enumerate(patients, start=1):
        t0 = time.perf_counter()
        await pipeline.run(patient, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)
        if idx % 10 == 0:
            print(f"  {idx}/{n} done  (last: {elapsed_ms:.1f} ms)")

    latencies_ms.sort()
    p50 = statistics.median(latencies_ms)
    p95 = latencies_ms[int(0.95 * len(latencies_ms))]
    p99 = latencies_ms[int(0.99 * len(latencies_ms))]
    mean = statistics.mean(latencies_ms)

    summary = {
        "n_queries": n,
        "top_k": top_k,
        "n_trials": n_trials,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "mean_ms": round(mean, 2),
        "min_ms": round(min(latencies_ms), 2),
        "max_ms": round(max(latencies_ms), 2),
    }

    print("\n" + "=" * 50)
    print("  TrialMate Latency Benchmark")
    print("=" * 50)
    print(f"  Queries        : {n}")
    print(f"  Trials indexed : {n_trials}")
    print(f"  top_k          : {top_k}")
    print(f"  p50            : {p50:.1f} ms")
    print(f"  p95            : {p95:.1f} ms")
    print(f"  p99            : {p99:.1f} ms")
    print(f"  mean           : {mean:.1f} ms")
    print("=" * 50)

    out_path = Path("evaluation/benchmark_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="TrialMate latency benchmark")
    parser.add_argument("--n", type=int, default=50, help="Number of patient queries")
    parser.add_argument("--top-k", type=int, default=5, help="top_k for each query")
    args = parser.parse_args()
    asyncio.run(run_benchmark(n=args.n, top_k=args.top_k))


if __name__ == "__main__":
    main()
