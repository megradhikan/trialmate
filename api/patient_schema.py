"""
api/patient_schema.py — Pydantic models for patient input
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PatientLabs(BaseModel):
    hemoglobin_a1c: Optional[float] = None
    creatinine: Optional[float] = None
    alt: Optional[float] = None
    ast: Optional[float] = None
    egfr: Optional[float] = None
    wbc: Optional[float] = None
    platelets: Optional[float] = None
    bilirubin: Optional[float] = None
    # Additional labs stored as arbitrary key-value
    model_config = {"extra": "allow"}


class PatientInput(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=120)
    sex: str = Field(..., pattern="^(male|female|other)$")
    conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    labs: PatientLabs = Field(default_factory=PatientLabs)
    notes: Optional[str] = None
    # FHIR extras (optional)
    icd_codes: list[str] = Field(default_factory=list)
    ecog_status: Optional[int] = Field(None, ge=0, le=4)
    is_pregnant: Optional[bool] = None
    prior_therapies: list[str] = Field(default_factory=list)

    def to_summary_text(self) -> str:
        """Convert patient to a natural language summary for embedding."""
        parts = [
            f"Age: {self.age}",
            f"Sex: {self.sex}",
        ]
        if self.conditions:
            parts.append(f"Conditions: {', '.join(self.conditions)}")
        if self.medications:
            parts.append(f"Medications: {', '.join(self.medications)}")
        lab_dict = self.labs.model_dump(exclude_none=True)
        if lab_dict:
            lab_str = "; ".join(f"{k}={v}" for k, v in lab_dict.items())
            parts.append(f"Labs: {lab_str}")
        if self.notes:
            parts.append(f"Notes: {self.notes}")
        return ". ".join(parts)
