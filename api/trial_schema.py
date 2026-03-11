"""
api/trial_schema.py — Pydantic models for trials and clauses
api/response_schema.py — Pydantic models for API responses
"""
from __future__ import annotations
from typing import Optional, Literal, Any
from pydantic import BaseModel, Field


# ─── Trial / Clause Models ──────────────────────────────────────────────────

class NormalizedConstraint(BaseModel):
    type: str  # age | sex | lab | condition | medication | exclusion | other
    # age
    min: Optional[float] = None
    max: Optional[float] = None
    # lab
    test: Optional[str] = None
    op: Optional[str] = None   # >=, <=, >, <, ==
    value: Optional[float] = None
    # sex / condition / medication
    allowed: Optional[list[str]] = None
    condition: Optional[str] = None
    medication: Optional[str] = None
    # exclusion catch-all
    text: Optional[str] = None


class TrialClause(BaseModel):
    nct_id: str
    clause_id: str
    text: str
    type: Literal["demographic", "diagnosis", "lab", "medication",
                  "procedure", "temporal", "location", "other", "mixed"]
    is_inclusion: bool = True   # True=inclusion, False=exclusion
    normalized: list[NormalizedConstraint] = Field(default_factory=list)
    embedding: Optional[list[float]] = None  # not serialized to API


class TrialMetadata(BaseModel):
    nct_id: str
    title: str
    phase: Optional[str] = None
    status: str = "Unknown"  # Recruiting | Not yet recruiting | Active, not recruiting | Terminated | Suspended
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    sponsor: Optional[str] = None
    url: Optional[str] = None
    clauses: list[TrialClause] = Field(default_factory=list)


# ─── Response Models ────────────────────────────────────────────────────────

class EvidenceCitation(BaseModel):
    source: str = "ClinicalTrials.gov"
    snippet: str
    url: Optional[str] = None


class MatchedClause(BaseModel):
    clause_id: str
    text: str
    type: str
    similarity: Optional[float] = None


class TrialMatch(BaseModel):
    nct_id: str
    title: str
    score: float = Field(..., ge=0.0, le=1.0)
    rule_score: float
    semantic_score: float
    trial_meta_score: float
    matched_clauses: list[MatchedClause] = Field(default_factory=list)
    violated_clauses: list[MatchedClause] = Field(default_factory=list)
    rationale: str
    confidence: Literal["low", "medium", "high"]
    evidence: list[EvidenceCitation] = Field(default_factory=list)


class MatchResponse(BaseModel):
    patient_id: str
    matches: list[TrialMatch]
    audit: Optional[dict[str, Any]] = None  # populated in debug mode


class PatientUploadResponse(BaseModel):
    patient_id: str
    status: str = "accepted"
    message: str = "Patient record stored. Call /patients/{id}/matches to retrieve matches."
