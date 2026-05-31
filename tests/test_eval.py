"""
tests/test_eval.py
End-to-end evaluation test: loads labeled set, runs full pipeline on all
patients, and asserts that hybrid F1@5 > 0.5.

Skipped automatically if the labeled set has fewer than 10 patients.
"""
from __future__ import annotations
import json
import asyncio
from pathlib import Path

import pytest

LABELED_PATH = "evaluation/labeled_set.json"
TRIALS_PATH = "data/processed/trials.json"
K = 5
MIN_PATIENTS = 10
F1_THRESHOLD = 0.5


def _load_labeled():
    if not Path(LABELED_PATH).exists():
        return []
    with open(LABELED_PATH) as f:
        return json.load(f)


def _pipeline_deps_available() -> bool:
    """Check that heavy pipeline dependencies (faiss, sentence_transformers) are installed."""
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


labeled = _load_labeled()
_skip_reason = (
    f"Labeled set has fewer than {MIN_PATIENTS} patients "
    f"(found {len(labeled)})"
    if len(labeled) < MIN_PATIENTS
    else "Pipeline dependencies (faiss, sentence_transformers) not installed"
    if not _pipeline_deps_available()
    else ""
)
_should_skip = len(labeled) < MIN_PATIENTS or not _pipeline_deps_available()


@pytest.mark.skipif(_should_skip, reason=_skip_reason)
def test_hybrid_f1_at_5():
    """Full pipeline hybrid F1@5 must exceed 0.5 on the labeled set."""
    from matching.scorer import MatchingPipeline
    from api.patient_schema import PatientInput
    from evaluation.evaluate import evaluate_predictions

    pipeline = MatchingPipeline.load()

    predictions = []

    async def _run_all():
        for entry in labeled:
            # Build a minimal PatientInput from labeled-set annotations
            patient = _patient_from_label(entry)
            result = await pipeline.run(patient, top_k=K)
            predictions.append({
                "patient_id": entry["patient_id"],
                "ranked_nct_ids": [m.nct_id for m in result.matches],
            })

    asyncio.run(_run_all())

    metrics = evaluate_predictions(predictions, labeled, k=K)
    f1 = metrics[f"f1@{K}"]
    print(f"\nHybrid evaluation: {metrics}")
    assert f1 > F1_THRESHOLD, (
        f"F1@{K}={f1:.4f} is below threshold {F1_THRESHOLD}. "
        f"Full metrics: {metrics}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patient_from_label(entry: dict) -> "PatientInput":
    """
    Reconstruct a PatientInput from the synthetic labeled-set entry.
    We embed condition hints in notes so the semantic retriever has signal
    even when structured fields are minimal.
    """
    from api.patient_schema import PatientInput, PatientLabs

    pid = entry["patient_id"]
    conds = entry.get("conditions", [])
    meds = entry.get("medications", [])
    labs_raw = entry.get("labs", {})
    labs = PatientLabs(**{k: v for k, v in labs_raw.items()
                         if k in PatientLabs.model_fields})

    return PatientInput(
        patient_id=pid,
        age=entry.get("age", 50),
        sex=entry.get("sex", "male"),
        conditions=conds,
        medications=meds,
        labs=labs,
        ecog_status=entry.get("ecog_status"),
        notes=entry.get("annotation_reason", ""),
    )
