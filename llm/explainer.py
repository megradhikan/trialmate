"""
llm/explainer.py
Generates patient-trial match explanations using the LLM.
"""
from __future__ import annotations
from loguru import logger

from api.patient_schema import PatientInput
from llm.prompts import (
    build_explanation_prompt,
    parse_explanation_output,
    EXPLANATION_SYSTEM,
)


async def generate_explanation(
    llm_client,
    patient: PatientInput,
    trial: dict,
    matched_clauses: list[dict],
    violated_clauses: list[dict],
    confidence: str,
) -> str:
    """
    Call LLM to generate a clinician-facing rationale.
    Returns plain string rationale (fallback to template on failure).
    """
    prompt = build_explanation_prompt(
        patient_summary=patient.to_summary_text(),
        nct_id=trial.get("nct_id", ""),
        trial_title=trial.get("title", ""),
        trial_status=trial.get("status", "Unknown"),
        trial_phase=trial.get("phase", "N/A"),
        matched_clauses=[
            {"clause_id": c.get("clause_id", ""), "text": c.get("text", "")[:200]}
            for c in matched_clauses[:5]
        ],
        violated_clauses=[
            {"clause_id": v.get("clause_id", ""), "text": str(v.get("reason", v.get("text", "")))[:200]}
            for v in violated_clauses[:5]
        ],
        confidence=confidence,
    )

    raw = await llm_client.complete(
        prompt=prompt,
        system=EXPLANATION_SYSTEM,
        max_tokens=600,
        temperature=0.1,
    )

    try:
        parsed = parse_explanation_output(raw)
        return parsed.get("rationale", raw)
    except Exception as e:
        logger.warning(f"Failed to parse explanation JSON: {e}")
        return raw[:500]  # Return raw truncated


async def split_clauses_with_llm(
    llm_client,
    nct_id: str,
    eligibility_text: str,
) -> list[dict]:
    """Use LLM to split eligibility text into atomic clauses."""
    from llm.prompts import build_splitter_prompt, parse_splitter_output, CLAUSE_SPLITTER_SYSTEM
    prompt = build_splitter_prompt(nct_id, eligibility_text)
    raw = await llm_client.complete(
        prompt=prompt,
        system=CLAUSE_SPLITTER_SYSTEM,
        max_tokens=2000,
        temperature=0.0,
    )
    return parse_splitter_output(raw)


async def normalize_clause_with_llm(
    llm_client,
    clause_text: str,
    is_inclusion: bool,
) -> dict | None:
    """Use LLM to extract a structured constraint from a clause."""
    from llm.prompts import build_normalizer_prompt, parse_normalizer_output, CLAUSE_NORMALIZER_SYSTEM
    prompt = build_normalizer_prompt(clause_text, is_inclusion)
    raw = await llm_client.complete(
        prompt=prompt,
        system=CLAUSE_NORMALIZER_SYSTEM,
        max_tokens=400,
        temperature=0.0,
    )
    return parse_normalizer_output(raw)
