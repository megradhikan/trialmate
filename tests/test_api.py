"""
tests/test_api.py
Integration tests for FastAPI endpoints using mocked pipeline.
"""
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


SAMPLE_PATIENT = {
    "patient_id": "P001",
    "age": 63,
    "sex": "male",
    "conditions": ["type 2 diabetes", "hypertension"],
    "medications": ["metformin"],
    "labs": {
        "hemoglobin_a1c": 8.2,
        "creatinine": 0.9,
    },
    "notes": "Poorly controlled diabetes, A1c 8.2%"
}

SAMPLE_MATCH_RESPONSE = {
    "patient_id": "P001",
    "matches": [
        {
            "nct_id": "NCT01234567",
            "title": "Study of Example Drug in T2DM",
            "score": 0.87,
            "rule_score": 1.0,
            "semantic_score": 0.82,
            "trial_meta_score": 1.0,
            "matched_clauses": [
                {"clause_id": "NCT01234567_C001", "text": "Ages 18 to 75 years", "type": "demographic"}
            ],
            "violated_clauses": [],
            "rationale": "Patient meets inclusion criteria. Confidence: high.",
            "confidence": "high",
            "evidence": [
                {"source": "ClinicalTrials.gov", "snippet": "Ages 18 to 75 years", "url": "https://clinicaltrials.gov/study/NCT01234567"}
            ]
        }
    ]
}


@pytest.fixture
def mock_pipeline():
    """Create a mock MatchingPipeline."""
    from api.trial_schema import MatchResponse, TrialMatch, MatchedClause, EvidenceCitation
    pipeline = MagicMock()
    pipeline.trials = {"NCT01234567": {"nct_id": "NCT01234567", "title": "Study of Example Drug in T2DM", "status": "Recruiting", "phase": "Phase 3", "clauses": []}}
    match_result = MatchResponse(**SAMPLE_MATCH_RESPONSE)
    pipeline.run = AsyncMock(return_value=match_result)
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create TestClient with mocked pipeline."""
    import api.main as main_module
    main_module._pipeline = mock_pipeline
    from api.main import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestPatientUpload:
    def test_upload_valid_patient(self, client):
        resp = client.post("/patients", json=SAMPLE_PATIENT)
        assert resp.status_code == 200
        assert resp.json()["patient_id"] == "P001"
        assert resp.json()["status"] == "accepted"

    def test_upload_invalid_age(self, client):
        bad = dict(SAMPLE_PATIENT, age=-1)
        resp = client.post("/patients", json=bad)
        assert resp.status_code == 422

    def test_upload_invalid_sex(self, client):
        bad = dict(SAMPLE_PATIENT, sex="unknown")
        resp = client.post("/patients", json=bad)
        assert resp.status_code == 422

    def test_upload_missing_required_field(self, client):
        bad = {k: v for k, v in SAMPLE_PATIENT.items() if k != "age"}
        resp = client.post("/patients", json=bad)
        assert resp.status_code == 422


class TestMatchEndpoint:
    def test_get_matches_uploaded_patient(self, client):
        # Upload first
        client.post("/patients", json=SAMPLE_PATIENT)
        # Then get matches
        resp = client.get("/patients/P001/matches?top_k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patient_id"] == "P001"
        assert "matches" in data
        assert len(data["matches"]) >= 1

    def test_get_matches_not_found(self, client):
        resp = client.get("/patients/DOESNOTEXIST/matches")
        assert resp.status_code == 404

    def test_get_matches_top_k_respected(self, client, mock_pipeline):
        client.post("/patients", json=SAMPLE_PATIENT)
        resp = client.get("/patients/P001/matches?top_k=3")
        assert resp.status_code == 200
        # Verify top_k was passed to pipeline
        call_kwargs = mock_pipeline.run.call_args
        assert call_kwargs.kwargs.get("top_k") == 3 or call_kwargs.args[1] == 3


class TestTrialEndpoint:
    def test_get_existing_trial(self, client):
        resp = client.get("/trials/NCT01234567")
        assert resp.status_code == 200
        assert resp.json()["nct_id"] == "NCT01234567"

    def test_get_missing_trial(self, client):
        resp = client.get("/trials/NCT99999999")
        assert resp.status_code == 404
