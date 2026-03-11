"""
llm/prompts.py
All TrialMate prompt templates. Ready-to-use with {placeholder} syntax.
Each template is paired with a parser function for its expected JSON output.
"""
from __future__ import annotations
import json
import re
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE 1: Clause Splitter & Atomicizer
# ═══════════════════════════════════════════════════════════════════════════

CLAUSE_SPLITTER_SYSTEM = """You are a clinical NLP expert. Your task is to split clinical trial eligibility text into atomic clauses. Return ONLY valid JSON — no markdown, no explanation."""

CLAUSE_SPLITTER_TEMPLATE = """Split the following clinical trial eligibility criteria into atomic, single-concept clauses.

RULES:
- Each clause should express exactly ONE eligibility criterion
- Separate inclusion and exclusion criteria
- Assign a type_hint from: demographic | diagnosis | lab | medication | procedure | temporal | location | other
- Preserve the original wording exactly
- Number clause_ids sequentially as {nct_id}_C001, {nct_id}_C002, etc.

TRIAL ID: {nct_id}

ELIGIBILITY TEXT:
{eligibility_text}

Return a JSON array with this exact schema:
[
  {{
    "clause_id": "string",
    "text": "exact clause text",
    "type_hint": "demographic|diagnosis|lab|medication|procedure|temporal|location|other",
    "is_inclusion": true or false
  }}
]

Return ONLY the JSON array. No preamble, no explanation."""

CLAUSE_SPLITTER_EXAMPLE_INPUT = {
    "nct_id": "NCT01234567",
    "eligibility_text": """Inclusion Criteria:
1. Ages 18 to 75 years
2. Confirmed diagnosis of type 2 diabetes mellitus
3. HbA1c between 7.0% and 10.0%
4. Currently on stable metformin therapy for at least 3 months

Exclusion Criteria:
1. History of congestive heart failure (NYHA Class III or IV)
2. Serum creatinine > 1.5 mg/dL
3. Pregnant or breastfeeding women"""
}

CLAUSE_SPLITTER_EXAMPLE_OUTPUT = [
    {"clause_id": "NCT01234567_C001", "text": "Ages 18 to 75 years", "type_hint": "demographic", "is_inclusion": True},
    {"clause_id": "NCT01234567_C002", "text": "Confirmed diagnosis of type 2 diabetes mellitus", "type_hint": "diagnosis", "is_inclusion": True},
    {"clause_id": "NCT01234567_C003", "text": "HbA1c between 7.0% and 10.0%", "type_hint": "lab", "is_inclusion": True},
    {"clause_id": "NCT01234567_C004", "text": "Currently on stable metformin therapy for at least 3 months", "type_hint": "medication", "is_inclusion": True},
    {"clause_id": "NCT01234567_C005", "text": "History of congestive heart failure (NYHA Class III or IV)", "type_hint": "diagnosis", "is_inclusion": False},
    {"clause_id": "NCT01234567_C006", "text": "Serum creatinine > 1.5 mg/dL", "type_hint": "lab", "is_inclusion": False},
    {"clause_id": "NCT01234567_C007", "text": "Pregnant or breastfeeding women", "type_hint": "demographic", "is_inclusion": False},
]


def build_splitter_prompt(nct_id: str, eligibility_text: str) -> str:
    return CLAUSE_SPLITTER_TEMPLATE.format(
        nct_id=nct_id,
        eligibility_text=eligibility_text,
    )


def parse_splitter_output(raw: str) -> list[dict]:
    """Parse LLM clause splitter output into list of clause dicts."""
    raw = raw.strip()
    # Strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE 2: Clause Normalizer / Structured Extractor
# ═══════════════════════════════════════════════════════════════════════════

CLAUSE_NORMALIZER_SYSTEM = """You are a clinical NLP expert specializing in structured data extraction from eligibility criteria. Return ONLY valid JSON."""

CLAUSE_NORMALIZER_TEMPLATE = """Extract a structured constraint from the following clinical trial eligibility clause.

PATIENT SCHEMA REFERENCE (for field names):
- age: integer (years)
- sex: "male" | "female" | "other"
- labs: {{ hemoglobin_a1c, creatinine, alt, ast, egfr, wbc, platelets, bilirubin }}
- conditions: list of strings (plain text condition names)
- medications: list of strings
- is_pregnant: boolean (optional)
- prior_therapies: list of strings
- ecog_status: integer 0-4 (optional)

TASK:
If the clause expresses a DETERMINISTIC constraint (age range, sex restriction, lab threshold, pregnancy requirement, medication requirement), return a JSON constraint object.
If the clause is non-deterministic (requires clinical judgment, complex history, etc.), return null and a label.

CLAUSE TEXT: "{clause_text}"
IS_INCLUSION: {is_inclusion}

Return ONE of these two formats:

Format A (deterministic constraint found):
{{
  "type": "age" | "sex" | "lab" | "condition" | "medication" | "exclusion",
  // For age: "min": number, "max": number
  // For sex: "allowed": ["male","female","other"]
  // For lab: "test": string, "op": ">=" | "<=" | ">" | "<" | "==", "value": number, "unit": string (optional)
  // For condition/medication: "value": string
  // For exclusion: "text": string
  "confidence": "high" | "medium" | "low"
}}

Format B (non-deterministic):
{{
  "structured": null,
  "label": "complex_history" | "functional_status" | "clinician_judgment" | "temporal" | "other",
  "reason": "one sentence explaining why deterministic extraction is not possible"
}}

Return ONLY the JSON object."""

CLAUSE_NORMALIZER_EXAMPLES = [
    {
        "input": {"clause_text": "Ages 18 to 75 years", "is_inclusion": True},
        "output": {"type": "age", "min": 18, "max": 75, "confidence": "high"}
    },
    {
        "input": {"clause_text": "HbA1c between 7.0% and 10.0%", "is_inclusion": True},
        "output": [
            {"type": "lab", "test": "hemoglobin_a1c", "op": ">=", "value": 7.0, "confidence": "high"},
            {"type": "lab", "test": "hemoglobin_a1c", "op": "<=", "value": 10.0, "confidence": "high"},
        ]
    },
    {
        "input": {"clause_text": "Female patients only", "is_inclusion": True},
        "output": {"type": "sex", "allowed": ["female"], "confidence": "high"}
    },
    {
        "input": {"clause_text": "No prior treatment with any GLP-1 receptor agonist in the last 6 months", "is_inclusion": False},
        "output": {"structured": None, "label": "temporal", "reason": "Requires knowledge of medication history timeline."}
    },
    {
        "input": {"clause_text": "ECOG performance status 0 or 1", "is_inclusion": True},
        "output": {"type": "lab", "test": "ecog_status", "op": "<=", "value": 1, "confidence": "high"}
    },
]


def build_normalizer_prompt(clause_text: str, is_inclusion: bool) -> str:
    return CLAUSE_NORMALIZER_TEMPLATE.format(
        clause_text=clause_text,
        is_inclusion=str(is_inclusion).lower(),
    )


def parse_normalizer_output(raw: str) -> dict | list | None:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════════════
# TEMPLATE 3: Explanation Generator
# ═══════════════════════════════════════════════════════════════════════════

EXPLANATION_SYSTEM = """You are a clinical research coordinator. Generate concise, clinician-facing rationales for clinical trial match results. Write in plain medical English. Return ONLY valid JSON."""

EXPLANATION_TEMPLATE = """Generate a structured explanation for why a patient does or does not match a clinical trial.

PATIENT SUMMARY:
{patient_summary}

TRIAL:
- NCT ID: {nct_id}
- Title: {trial_title}
- Status: {trial_status}
- Phase: {trial_phase}

MATCHED CLAUSES (criteria the patient appears to meet):
{matched_clauses_json}

VIOLATED CLAUSES (criteria the patient may not meet or that require review):
{violated_clauses_json}

PRELIMINARY CONFIDENCE: {confidence}

TASK: Write a 2-4 sentence clinician-facing rationale that:
1. States whether the patient likely qualifies, cites specific clause_ids
2. Lists matched criteria with brief values
3. Notes any uncertain or violated criteria requiring clinician review
4. Ends with a confidence label

Return this JSON:
{{
  "rationale": "2-4 sentence plain-language explanation citing clause_ids",
  "confidence": "low" | "medium" | "high",
  "evidence_bullets": [
    "Criterion X (clause_id: C001): patient value meets threshold",
    ...
  ],
  "clinician_review_items": [
    "Item requiring manual review",
    ...
  ]
}}

Return ONLY the JSON object."""

EXPLANATION_EXAMPLE_INPUT = {
    "patient_summary": "Age: 63. Sex: male. Conditions: type 2 diabetes, hypertension. Medications: metformin. Labs: hemoglobin_a1c=8.2; creatinine=0.9.",
    "nct_id": "NCT01234567",
    "trial_title": "Study of Example Drug in Type 2 Diabetes Mellitus",
    "trial_status": "Recruiting",
    "trial_phase": "Phase 3",
    "matched_clauses": [
        {"clause_id": "NCT01234567_C001", "text": "Ages 18 to 75 years"},
        {"clause_id": "NCT01234567_C002", "text": "Confirmed diagnosis of type 2 diabetes mellitus"},
        {"clause_id": "NCT01234567_C003", "text": "HbA1c between 7.0% and 10.0%"},
    ],
    "violated_clauses": [],
    "confidence": "high",
}

EXPLANATION_EXAMPLE_OUTPUT = {
    "rationale": "This 63-year-old male with type 2 diabetes meets the primary inclusion criteria for NCT01234567. His HbA1c of 8.2% satisfies the 7–10% range (C003), he has a confirmed T2DM diagnosis (C002), and his age is within the 18–75 range (C001). No hard exclusions were identified; however, CHF history (C005) should be verified by the site clinician before enrollment.",
    "confidence": "high",
    "evidence_bullets": [
        "Age 63 meets criterion C001 (Ages 18–75)",
        "Confirmed T2DM satisfies C002",
        "HbA1c 8.2% is within required range 7.0–10.0% (C003)",
        "Creatinine 0.9 mg/dL is below typical exclusion thresholds",
    ],
    "clinician_review_items": [
        "Verify absence of CHF Class III/IV (C005 — exclusion criterion, not in patient record)",
        "Confirm metformin has been stable for ≥3 months (C004)",
    ]
}


def build_explanation_prompt(
    patient_summary: str,
    nct_id: str,
    trial_title: str,
    trial_status: str,
    trial_phase: str,
    matched_clauses: list[dict],
    violated_clauses: list[dict],
    confidence: str,
) -> str:
    return EXPLANATION_TEMPLATE.format(
        patient_summary=patient_summary,
        nct_id=nct_id,
        trial_title=trial_title,
        trial_status=trial_status or "Unknown",
        trial_phase=trial_phase or "N/A",
        matched_clauses_json=json.dumps(matched_clauses, indent=2),
        violated_clauses_json=json.dumps(violated_clauses, indent=2),
        confidence=confidence,
    )


def parse_explanation_output(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)
