"""
evaluation/run_ablation.py
Reproducible ablation of the TrialMate ranking modes on the labeled set.

Modes (all rank every indexed trial for each patient):
  rule_only      rule_score only (ties broken by a seeded shuffle)
  semantic_only  mean inclusion-clause similarity only
  hybrid         fixed 0.6/0.3/0.1 weighted formula (hard exclusions -> 0)
  learned        logistic regression on the same 3 features, K-fold
                 cross-validated BY PATIENT (never scored on its own training data)

Two label sources (--labels):
  registry  (default) silver labels from ClinicalTrials.gov structured fields
            (condition + age + sex) for 51 generated patients, ~71 relevant
            trials each.  See evaluation/build_registry_labels.py.
  author    50 author-written labels, ~2 relevant trials each.  Incomplete by
            construction (many equally valid trials are unlabeled), so absolute
            precision is capped and noisy; kept for transparency.

Patients are built from structured fields only; free-text annotator notes are
NOT given to the retriever (they would leak the labels).

Usage:
    python -m evaluation.run_ablation --model all-MiniLM-L6-v2
    python -m evaluation.run_ablation --model pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
    python -m evaluation.run_ablation --train-reranker   # fit on all patients, write reranker.json
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model.split("/")[-1]).strip("_").lower()


def patient_from_label(entry: dict):
    from api.patient_schema import PatientInput, PatientLabs

    labs = PatientLabs(**{k: v for k, v in entry.get("labs", {}).items() if k in PatientLabs.model_fields})
    return PatientInput(
        patient_id=entry["patient_id"],
        age=entry.get("age", 50),
        sex=entry.get("sex", "male"),
        conditions=entry.get("conditions", []),
        medications=entry.get("medications", []),
        labs=labs,
        ecog_status=entry.get("ecog_status"),
    )


METRICS = ["p@5", "r@5", "p@10", "ndcg@10", "mrr"]


def metrics(ranked: list[str], relevant: set[str]) -> list[float]:
    hits = [1 if n in relevant else 0 for n in ranked[:10]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
    rr = next((1.0 / (i + 1) for i, n in enumerate(ranked) if n in relevant), 0.0)
    return [sum(hits[:5]) / 5, sum(hits[:5]) / len(relevant), sum(hits) / 10, dcg / ideal, rr]


def summarize(per_patient: list[list[float]], n_boot: int = 2000, seed: int = 0) -> dict:
    a = np.array(per_patient)
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(a), len(a)) for _ in range(n_boot)]
    out = {m: round(float(a[:, i].mean()), 3) for i, m in enumerate(METRICS)}
    for m in ("p@5", "ndcg@10"):
        i = METRICS.index(m)
        boots = [a[ix, i].mean() for ix in idx]
        out[f"{m}_ci95"] = [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="sentence-transformers model (sets EMBEDDING_MODEL)")
    ap.add_argument("--labels", choices=["registry", "author"], default="registry")
    ap.add_argument("--trials", default="data/processed/trials.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-reranker", action="store_true")
    args = ap.parse_args()

    if args.model:
        os.environ["EMBEDDING_MODEL"] = args.model
    os.environ.setdefault("LLM_PROVIDER", "mock")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from matching import embedder
    from matching.scorer import MatchingPipeline, LogisticRegressionRanker

    model = embedder.EMBEDDING_MODEL
    index_dir = f"data/processed/faiss_index_{_slug(model)}"
    pipeline = MatchingPipeline.load(trials_path=args.trials, index_dir=index_dir,
                                     reranker_path="/nonexistent/reranker.json")
    pipeline.reranker = None  # ablation uses explicit scoring per mode

    known = set(pipeline.trials)
    if args.labels == "author":
        labeled = json.load(open("evaluation/labeled_set.json"))
        rel = {e["patient_id"]: set(e["eligible_nct_ids"]) & known for e in labeled}
    else:
        by_id = {p["patient_id"]: p for p in json.load(open("data/samples/patients.json"))}
        labeled = []
        rel = {}
        for e in json.load(open("evaluation/registry_labels.json")):
            labeled.append(by_id[e["patient_id"]])
            rel[e["patient_id"]] = set(e["relevant_nct_ids"]) & known
    labeled = [e for e in labeled if rel[e["patient_id"]]]

    # Features for every (patient, trial) pair
    feats: dict[str, list[dict]] = {}
    for e in labeled:
        cands, _ = pipeline.candidate_features(patient_from_label(e), all_trials=True)
        feats[e["patient_id"]] = [
            {"nct_id": c["trial"]["nct_id"], "rule": c["rule_score"], "sem": c["semantic_score"],
             "meta": c["trial_meta_score"], "fixed": c["fixed_score"]}
            for c in cands
        ]

    rng = random.Random(args.seed)

    def rank_by(pid, key):
        rows = feats[pid][:]
        rng.shuffle(rows)  # random tie-break
        rows.sort(key=lambda r: key(r), reverse=True)
        return [r["nct_id"] for r in rows]

    results: dict[str, list] = {m: [] for m in ["rule_only", "semantic_only", "hybrid", "learned", "random"]}
    pids = [e["patient_id"] for e in labeled]
    for pid in pids:
        R = rel[pid]
        results["random"].append(metrics(rank_by(pid, lambda r: 0), R))
        results["rule_only"].append(metrics(rank_by(pid, lambda r: r["rule"]), R))
        results["semantic_only"].append(metrics(rank_by(pid, lambda r: r["sem"]), R))
        results["hybrid"].append(metrics(rank_by(pid, lambda r: r["fixed"]), R))

    def xy(ps):
        X, y = [], []
        for pid in ps:
            for r in feats[pid]:
                X.append([r["rule"], r["sem"], r["meta"]])
                y.append(int(r["nct_id"] in rel[pid]))
        return X, y

    # Learned reranker: K-fold CV by patient
    order = pids[:]
    random.Random(args.seed).shuffle(order)
    folds = [order[i::args.folds] for i in range(args.folds)]
    learned_by_pid = {}
    for i, test in enumerate(folds):
        train = [p for j, f in enumerate(folds) if j != i for p in f]
        ranker = LogisticRegressionRanker().fit(*xy(train))
        for pid in test:
            rows = feats[pid][:]
            rng.shuffle(rows)
            rows.sort(key=lambda r: ranker.predict_score(r["rule"], r["sem"], r["meta"]), reverse=True)
            learned_by_pid[pid] = metrics([r["nct_id"] for r in rows], rel[pid])
    results["learned"] = [learned_by_pid[p] for p in pids]

    summary = {m: summarize(v) for m, v in results.items()}
    full = LogisticRegressionRanker().fit(*xy(pids))
    sizes = [len(rel[p]) for p in pids]
    out = {
        "embedding_model": model,
        "label_source": args.labels,
        "n_patients": len(pids),
        "n_trials": len(pipeline.trials),
        "relevant_per_patient": {"mean": round(float(np.mean(sizes)), 1), "min": min(sizes), "max": max(sizes)},
        "results": summary,
        "reranker_standardized_coefficients": {k: round(v, 3) for k, v in full.learned_weights().items()},
    }
    out_path = args.out or f"evaluation/results_{args.labels}_{_slug(model)}.json"
    Path(out_path).write_text(json.dumps(out, indent=2) + "\n")

    print(f"\nmodel={model}  labels={args.labels}  patients={len(pids)}  trials={len(pipeline.trials)}  "
          f"relevant/patient={out['relevant_per_patient']}")
    print(f"{'mode':15s} {'P@5':>6s} {'R@5':>6s} {'P@10':>6s} {'nDCG@10':>8s} {'  nDCG 95% CI':>16s} {'MRR':>6s}")
    for m, s_ in summary.items():
        print(f"{m:15s} {s_['p@5']:6.3f} {s_['r@5']:6.3f} {s_['p@10']:6.3f} {s_['ndcg@10']:8.3f} "
              f"{str(s_['ndcg@10_ci95']):>16s} {s_['mrr']:6.3f}")
    print("coefficients (standardized):", out["reranker_standardized_coefficients"])

    if args.train_reranker:
        full.save()
        print("saved data/processed/reranker.json")


if __name__ == "__main__":
    main()
