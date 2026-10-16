"""Unit tests for scoring, the learned reranker, and pipeline candidate generation."""
from __future__ import annotations
import asyncio

import pytest

from api.patient_schema import PatientInput
from matching.scorer import (
    LogisticRegressionRanker,
    MatchingPipeline,
    compute_semantic_score,
    compute_trial_meta_score,
    score_trial,
)

RECRUITING_P3 = {"nct_id": "NCT1", "status": "Recruiting", "phase": "Phase 3"}
CLOSED = {"nct_id": "NCT2", "status": "Completed", "phase": "N/A"}


class TestScoring:
    def test_semantic_score_is_mean_clamped(self):
        assert compute_semantic_score([]) == 0.0
        assert compute_semantic_score([0.2, 0.4]) == pytest.approx(0.3)
        assert compute_semantic_score([2.0]) == 1.0
        assert compute_semantic_score([-1.0]) == 0.0

    def test_meta_score_prefers_recruiting_late_phase(self):
        assert compute_trial_meta_score(RECRUITING_P3) > compute_trial_meta_score(CLOSED)

    def test_hard_exclusion_zeroes_score(self):
        assert score_trial(RECRUITING_P3, rule_score=0.0, clause_sims=[0.99]) == 0.0

    def test_score_monotonic_in_rule_and_semantics(self):
        base = score_trial(RECRUITING_P3, 0.5, [0.5])
        assert score_trial(RECRUITING_P3, 1.0, [0.5]) > base
        assert score_trial(RECRUITING_P3, 0.5, [0.9]) > base

    def test_score_bounded(self):
        s = score_trial(RECRUITING_P3, 1.0, [1.0], w_rule=5, w_sem=5, w_trial=5)
        assert 0.0 <= s <= 1.0


class TestReranker:
    def _data(self):
        # rule_score drives the label; semantic is noise-free but weaker
        X = [[1.0, 0.8, 0.9], [1.0, 0.6, 0.5], [0.0, 0.7, 0.9], [0.5, 0.2, 0.3],
             [1.0, 0.9, 0.7], [0.0, 0.1, 0.2], [0.5, 0.5, 0.5], [1.0, 0.4, 0.6]]
        y = [1, 1, 0, 0, 1, 0, 0, 1]
        return X, y

    def test_untrained_raises(self):
        with pytest.raises(RuntimeError):
            LogisticRegressionRanker().predict_score(1, 1, 1)

    def test_fit_requires_both_classes(self):
        with pytest.raises(ValueError):
            LogisticRegressionRanker().fit([[1, 1, 1], [0, 0, 0]], [1, 1])

    def test_fit_predicts_in_unit_interval_and_ranks(self):
        r = LogisticRegressionRanker().fit(*self._data())
        hi, lo = r.predict_score(1.0, 0.8, 0.9), r.predict_score(0.0, 0.1, 0.2)
        assert 0.0 <= lo < hi <= 1.0

    def test_json_roundtrip(self, tmp_path):
        r = LogisticRegressionRanker().fit(*self._data())
        path = tmp_path / "reranker.json"
        r.save(str(path))
        r2 = LogisticRegressionRanker.load(str(path))
        assert r2.is_trained
        assert r2.predict_score(1.0, 0.5, 0.5) == r.predict_score(1.0, 0.5, 0.5)

    def test_load_missing_file_is_untrained(self, tmp_path):
        assert not LogisticRegressionRanker.load(str(tmp_path / "nope.json")).is_trained


class _FakeIndex:
    """Stands in for ClauseIndex: returns fixed retrieved clauses."""

    def __init__(self, hits):
        self.hits = hits
        self.index = None

    def search(self, query_text, top_k=20):
        return [dict(h) for h in self.hits]


def _trial(nct, clauses, status="Recruiting"):
    return {"nct_id": nct, "title": f"Trial {nct}", "status": status, "phase": "Phase 3",
            "clauses": [{"clause_id": f"{nct}-{i}", "text": t, "is_inclusion": inc, "type": "other"}
                        for i, (t, inc) in enumerate(clauses)]}


@pytest.fixture
def pipeline():
    trials = {
        "NCT_OK": _trial("NCT_OK", [("Age 18 years or older", True)]),
        "NCT_OLD": _trial("NCT_OLD", [("Age 18 to 40 years", True)]),
        "NCT_OTHER": _trial("NCT_OTHER", [("Unrelated criterion", True)]),
    }
    hits = [
        {"nct_id": "NCT_OK", "clause_id": "NCT_OK-0", "text": "Age 18 years or older", "similarity": 0.8, "is_inclusion": True, "type": "age"},
        {"nct_id": "NCT_OLD", "clause_id": "NCT_OLD-0", "text": "Age 18 to 40 years", "similarity": 0.9, "is_inclusion": True, "type": "age"},
    ]
    return MatchingPipeline(trials=trials, clause_index=_FakeIndex(hits), llm_client=None, reranker=None)


def _patient(age=65):
    return PatientInput(patient_id="T1", age=age, sex="male", conditions=["hypertension"])


class TestPipeline:
    def test_age_violation_zeroes_that_trial(self, pipeline):
        cands, n = pipeline.candidate_features(_patient(65))
        by = {c["trial"]["nct_id"]: c for c in cands}
        assert n == 2
        assert by["NCT_OLD"]["rule_score"] == 0.0 and by["NCT_OLD"]["score"] == 0.0
        assert by["NCT_OK"]["rule_score"] == 1.0 and by["NCT_OK"]["score"] > 0.0

    def test_all_trials_flag_includes_unretrieved(self, pipeline):
        cands, _ = pipeline.candidate_features(_patient(), all_trials=True)
        assert {c["trial"]["nct_id"] for c in cands} == {"NCT_OK", "NCT_OLD", "NCT_OTHER"}

    def test_run_ranks_eligible_first_with_fallback_rationale(self, pipeline):
        res = asyncio.run(pipeline.run(_patient(65), top_k=3))
        assert res.matches[0].nct_id == "NCT_OK"
        assert "Clinician review required" in res.matches[0].rationale
        scores = [m.score for m in res.matches]
        assert scores == sorted(scores, reverse=True)

    def test_young_patient_matches_both_age_trials(self, pipeline):
        cands, _ = pipeline.candidate_features(_patient(30))
        assert all(c["rule_score"] == 1.0 for c in cands if c["trial"]["nct_id"] in {"NCT_OK", "NCT_OLD"})
