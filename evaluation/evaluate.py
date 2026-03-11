"""
evaluation/evaluate.py
Compute Precision@k, Recall@k, F1 for TrialMate.
Also supports ablation: semantic-only, rule-only, hybrid.

Usage:
    python evaluation/evaluate.py \
        --labeled data/labeled_set.json \
        --trials data/processed/trials.json \
        --mode hybrid
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def f1_at_k(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_predictions(
    predictions: list[dict],  # [{patient_id, ranked_nct_ids: [...]}]
    labels: list[dict],       # [{patient_id, eligible_nct_ids: [...]}]
    k: int = 5,
) -> dict:
    label_map = {l["patient_id"]: set(l.get("eligible_nct_ids", [])) for l in labels}
    ps, rs = [], []
    for pred in predictions:
        pid = pred["patient_id"]
        retrieved = pred.get("ranked_nct_ids", [])
        relevant = label_map.get(pid, set())
        if not relevant:
            continue
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        ps.append(p)
        rs.append(r)

    mean_p = sum(ps) / len(ps) if ps else 0.0
    mean_r = sum(rs) / len(rs) if rs else 0.0
    mean_f1 = f1_at_k(mean_p, mean_r)
    return {
        f"precision@{k}": round(mean_p, 4),
        f"recall@{k}": round(mean_r, 4),
        f"f1@{k}": round(mean_f1, 4),
        "n_patients": len(ps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", default="evaluation/labeled_set.json")
    parser.add_argument("--predictions", default="evaluation/predictions.json",
                        help="JSON file with model predictions per patient")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "semantic_only", "rule_only"], default="hybrid")
    args = parser.parse_args()

    with open(args.labeled) as f:
        labels = json.load(f)

    if not Path(args.predictions).exists():
        print(f"No predictions file at {args.predictions}. Run the API first to generate predictions.")
        return

    with open(args.predictions) as f:
        predictions = json.load(f)

    metrics = evaluate_predictions(predictions, labels, k=args.k)
    print(f"\n=== Evaluation Results (mode={args.mode}, k={args.k}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
