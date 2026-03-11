"""
ingestion/trial_ingestor.py
Downloads/reads ClinicalTrials.gov JSON, extracts trial fields, 
and produces normalized JSON with clause metadata.

Usage:
    python ingestion/trial_ingestor.py \
        --input data/raw/sample_trials.json \
        --output data/processed/trials.json \
        --seed 42
"""
from __future__ import annotations
import argparse
import json
import re
import time
import random
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger
from tqdm import tqdm


CTGOV_SEARCH_URL = (
    "https://clinicaltrials.gov/api/v2/studies"
    "?query.cond={condition}&pageSize={page_size}&format=json"
)

# Clause type classification keywords
CLAUSE_TYPE_KEYWORDS = {
    "demographic": ["age", "sex", "male", "female", "gender", "years old", "adult", "pediatric"],
    "diagnosis": ["diagnosis", "diagnosed", "history of", "confirmed", "disease", "cancer", "diabetes", "hypertension"],
    "lab": ["hba1c", "creatinine", "alt", "ast", "egfr", "hemoglobin", "platelet", "bilirubin", "wbc", "lab"],
    "medication": ["taking", "receiving", "treatment with", "prior therapy", "metformin", "insulin", "chemotherapy"],
    "procedure": ["surgery", "transplant", "biopsy", "procedure", "operation"],
    "temporal": ["within", "months", "weeks", "years prior", "recent", "current", "ongoing"],
    "location": ["site", "center", "country", "geographic"],
}


def classify_clause_type(text: str) -> str:
    text_lower = text.lower()
    scores = {ctype: 0 for ctype in CLAUSE_TYPE_KEYWORDS}
    for ctype, keywords in CLAUSE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[ctype] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


def split_eligibility_text(text: str, nct_id: str) -> list[dict]:
    """
    Simple regex-based clause splitter (fallback when LLM not available).
    Splits on sentence boundaries and numbered lists.
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Detect inclusion/exclusion sections
    inc_match = re.search(r'inclusion criteria[:\s]*(.*?)(?=exclusion criteria|$)', text, re.IGNORECASE | re.DOTALL)
    exc_match = re.search(r'exclusion criteria[:\s]*(.*?)$', text, re.IGNORECASE | re.DOTALL)

    clauses = []

    def _split_section(section_text: str, is_inclusion: bool) -> list[dict]:
        # Split on numbered items or semicolons or newlines
        items = re.split(r'(?:\d+\.\s+|\n|;\s*)', section_text)
        result = []
        for i, item in enumerate(items):
            item = item.strip()
            if len(item) < 10:
                continue
            clause_id = f"{nct_id}_{'I' if is_inclusion else 'E'}{i+1:03d}"
            result.append({
                "nct_id": nct_id,
                "clause_id": clause_id,
                "text": item,
                "type": classify_clause_type(item),
                "is_inclusion": is_inclusion,
                "normalized": [],
            })
        return result

    if inc_match:
        clauses.extend(_split_section(inc_match.group(1), is_inclusion=True))
    if exc_match:
        clauses.extend(_split_section(exc_match.group(1), is_inclusion=False))
    if not clauses:
        # No section headers found — treat entire text as inclusion
        clauses.extend(_split_section(text, is_inclusion=True))

    return clauses


def parse_trial_from_simplified(study: dict) -> Optional[dict]:
    """Handle our own pre-parsed simplified trial format (already has nct_id, clauses, etc.)."""
    if "nct_id" not in study:
        return None
    # If clauses are missing or empty, re-split from eligibility_text
    if not study.get("clauses") and study.get("eligibility_text"):
        study["clauses"] = split_eligibility_text(study["eligibility_text"], study["nct_id"])
    return study


def parse_trial_from_ctgov_v2(study: dict) -> Optional[dict]:
    """Parse a ClinicalTrials.gov v2 API study object into our schema."""
    try:
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        eligibility = proto.get("eligibilityModule", {})
        conditions_mod = proto.get("conditionsModule", {})
        interventions_mod = proto.get("armsInterventionsModule", {})
        desc = proto.get("descriptionModule", {})

        nct_id = ident.get("nctId", "")
        if not nct_id:
            return None

        title = ident.get("briefTitle", "")
        phase = design.get("phases", [""])[0] if design.get("phases") else None
        status = status_mod.get("overallStatus", "Unknown")
        eligibility_text = eligibility.get("eligibilityCriteria", "")
        conditions = conditions_mod.get("conditions", [])
        interventions = [
            i.get("name", "") for i in interventions_mod.get("interventions", [])
        ]
        sponsor = proto.get("sponsorCollaboratorsModule", {}).get(
            "leadSponsor", {}
        ).get("name", "")

        clauses = split_eligibility_text(eligibility_text, nct_id)

        return {
            "nct_id": nct_id,
            "title": title,
            "phase": phase,
            "status": status,
            "conditions": conditions,
            "interventions": interventions,
            "sponsor": sponsor,
            "url": f"https://clinicaltrials.gov/study/{nct_id}",
            "eligibility_text": eligibility_text,
            "clauses": clauses,
        }
    except Exception as e:
        logger.warning(f"Failed to parse trial: {e}")
        return None


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

def download_trials(conditions: list[str], page_size: int = 100) -> list[dict]:
    """Download trials from ClinicalTrials.gov API v2."""
    all_studies = []
    for condition in conditions:
        url = CTGOV_SEARCH_URL.format(condition=condition.replace(" ", "+"), page_size=page_size)
        logger.info(f"Downloading trials for: {condition}")
        try:
            resp = httpx.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
            studies = data.get("studies", [])
            logger.info(f"  Got {len(studies)} studies")
            all_studies.extend(studies)
            time.sleep(0.5)  # rate limit
        except Exception as e:
            logger.error(f"  Failed to download: {e}")
    return all_studies


def ingest_from_file(input_path: str) -> list[dict]:
    """Load trials from a local JSON file."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        # Already a list of study objects
        return data
    if isinstance(data, dict) and "studies" in data:
        return data["studies"]
    raise ValueError(f"Unexpected JSON structure in {input_path}")


def main():
    parser = argparse.ArgumentParser(description="TrialMate Trial Ingestor")
    parser.add_argument("--input", help="Local JSON file (ClinicalTrials.gov v2 format)")
    parser.add_argument("--conditions", nargs="+", default=["type 2 diabetes", "hypertension"],
                        help="Conditions to search if no input file")
    parser.add_argument("--output", default="data/processed/trials.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.input:
        logger.info(f"Reading trials from {args.input}")
        raw_studies = ingest_from_file(args.input)
    else:
        logger.info("Downloading trials from ClinicalTrials.gov API")
        raw_studies = download_trials(args.conditions)

    logger.info(f"Parsing {len(raw_studies)} raw studies...")
    trials = []
    for study in tqdm(raw_studies):
        # Auto-detect format: if study already has nct_id at top level, it's our simplified format
        if "nct_id" in study:
            parsed = parse_trial_from_simplified(study)
        else:
            parsed = parse_trial_from_ctgov_v2(study)
        if parsed:
            trials.append(parsed)

    logger.info(f"Parsed {len(trials)} valid trials.")
    with open(args.output, "w") as f:
        json.dump(trials, f, indent=2)
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()