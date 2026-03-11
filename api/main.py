"""
api/main.py — FastAPI application for TrialMate
"""
from __future__ import annotations
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.patient_schema import PatientInput
from api.trial_schema import (
    PatientUploadResponse, MatchResponse, TrialMetadata,
)
from matching.scorer import MatchingPipeline

# ─── In-memory session store (no PHI persisted) ─────────────────────────────
_patient_store: dict[str, PatientInput] = {}
_pipeline: Optional[MatchingPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup."""
    global _pipeline
    logger.info("Loading TrialMate matching pipeline...")
    _pipeline = MatchingPipeline.load()
    logger.info(f"Pipeline ready. {len(_pipeline.trials)} trials indexed.")
    yield
    logger.info("Shutting down — clearing patient session store.")
    _patient_store.clear()


app = FastAPI(
    title="TrialMate — Clinical Trial Matching Agent",
    description=(
        "⚠️ PROTOTYPE ONLY. NOT FOR CLINICAL USE. "
        "Matches synthetic patient records to ClinicalTrials.gov entries."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "trials_loaded": len(_pipeline.trials) if _pipeline else 0}


@app.post("/patients", response_model=PatientUploadResponse)
async def upload_patient(patient: PatientInput):
    """
    Upload a synthetic patient JSON record.
    Returns a patient_id for subsequent match queries.
    NOTE: records are stored in-memory only and deleted on server shutdown.
    """
    _patient_store[patient.patient_id] = patient
    logger.info(f"Patient {patient.patient_id} uploaded (in-memory only).")
    return PatientUploadResponse(patient_id=patient.patient_id)


@app.get("/patients/{patient_id}/matches", response_model=MatchResponse)
async def get_matches(
    patient_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
    debug: bool = Query(default=False),
):
    """
    Retrieve top-k ranked clinical trial matches for a patient.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")
    patient = _patient_store.get(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")

    result = await _pipeline.run(patient, top_k=top_k, debug=debug)

    # PHI hygiene: remove patient from store after serving results
    # (comment out if you want to allow multiple queries per session)
    # del _patient_store[patient_id]

    return result


@app.get("/trials/{nct_id}", response_model=TrialMetadata)
async def get_trial(nct_id: str):
    """Return trial metadata and clauses."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")
    trial = _pipeline.trials.get(nct_id)
    if trial is None:
        raise HTTPException(status_code=404, detail=f"Trial {nct_id} not found.")
    return trial
