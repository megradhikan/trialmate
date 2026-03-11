"""
matching/scorer.py
Combines rule checks + semantic similarity + trial metadata into a 0–1 match score.
Also houses the full MatchingPipeline.

Scoring formula:
    score = w_rule * rule_score + w_sem * semantic_score + w_trial * trial_meta_score

Default weights: w_rule=0.6, w_sem=0.3, w_trial=0.1
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger

from api.patient_schema import PatientInput
from api.trial_schema import MatchResponse, TrialMatch, MatchedClause, EvidenceCitation
from matching.clause_normalizer import normalize_clause, verify_constraints
from matching.embedder import ClauseIndex

# ─── Scoring weights (editable via env) ─────────────────────────────────────
W_RULE = float(os.getenv("SCORING_W_RULE", "0.6"))
W_SEM = float(os.getenv("SCORING_W_SEM", "0.3"))
W_TRIAL = float(os.getenv("SCORING_W_TRIAL", "0.1"))


STATUS_META_SCORES = {
    "Recruiting": 1.0,
    "Active, not recruiting": 0.5,
    "Not yet recruiting": 0.7,
    "Enrolling by invitation": 0.6,
    "Completed": 0.0,
    "Suspended": 0.0,
    "Terminated": 0.0,
    "Withdrawn": 0.0,
    "Unknown": 0.3,
}

PHASE_META_SCORES = {
    "Phase 3": 1.0,
    "Phase 2": 0.9,
    "Phase 2/Phase 3": 0.95,
    "Phase 1/Phase 2": 0.7,
    "Phase 1": 0.5,
    "Phase 4": 1.0,
    "Early Phase 1": 0.3,
    "N/A": 0.5,
}


def compute_trial_meta_score(trial: dict) -> float:
    status_score = STATUS_META_SCORES.get(trial.get("status", "Unknown"), 0.3)
    phase_score = PHASE_META_SCORES.get(trial.get("phase", "N/A"), 0.5)
    return 0.7 * status_score + 0.3 * phase_score


def compute_semantic_score(clause_similarities: list[float]) -> float:
    """Mean cosine similarity from inclusion clause retrievals, 0–1."""
    if not clause_similarities:
        return 0.0
    return min(1.0, max(0.0, sum(clause_similarities) / len(clause_similarities)))


def score_trial(
    trial: dict,
    rule_score: float,
    clause_sims: list[float],
    w_rule: float = W_RULE,
    w_sem: float = W_SEM,
    w_trial: float = W_TRIAL,
) -> float:
    sem = compute_semantic_score(clause_sims)
    meta = compute_trial_meta_score(trial)
    total = w_rule * rule_score + w_sem * sem + w_trial * meta
    return round(min(1.0, max(0.0, total)), 4)


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class MatchingPipeline:
    def __init__(self, trials: dict, clause_index: ClauseIndex, llm_client=None):
        self.trials = trials          # {nct_id: trial_dict}
        self.clause_index = clause_index
        self.llm_client = llm_client  # optional; used for explanations

    @classmethod
    def load(
        cls,
        trials_path: str = "data/processed/trials.json",
        index_dir: str = "data/processed/faiss_index",
    ) -> "MatchingPipeline":
        # Load trials
        if Path(trials_path).exists():
            with open(trials_path) as f:
                trial_list = json.load(f)
        else:
            logger.warning(f"No trials file at {trials_path}. Using empty set.")
            trial_list = []

        trials_by_id = {t["nct_id"]: t for t in trial_list}

        # Load or build clause index
        # Rebuild if trials file is newer than the index (catches live data updates)
        idx = ClauseIndex()
        index_faiss = Path(index_dir) / "clauses.faiss"
        trials_mtime = Path(trials_path).stat().st_mtime if Path(trials_path).exists() else 0
        index_mtime = index_faiss.stat().st_mtime if index_faiss.exists() else 0
        index_stale = not index_faiss.exists() or trials_mtime > index_mtime

        if index_stale:
            logger.info(f"Building FAISS index for {len(trial_list)} trials (index missing or stale)...")
            idx.build(trial_list, output_dir=index_dir)
        else:
            idx.load(index_dir)

        logger.info(f"Trials loaded: {len(trials_by_id)} | Index vectors: {idx.index.ntotal if idx.index else 0}")

        # Try to load LLM client
        try:
            from llm.client import LLMClient
            llm = LLMClient()
        except Exception as e:
            logger.warning(f"LLM client not available: {e}")
            llm = None

        return cls(trials=trials_by_id, clause_index=idx, llm_client=llm)

    async def run(
        self,
        patient: PatientInput,
        top_k: int = 5,
        debug: bool = False,
    ) -> MatchResponse:
        patient_summary = patient.to_summary_text()
        logger.debug(f"Patient summary: {patient_summary}")

        # 1. Retrieve top clause candidates — use larger pool for bigger trial sets
        n_trials = len(self.trials)
        retrieve_n = max(top_k * 20, min(n_trials * 5, 500))
        retrieved = self.clause_index.search(patient_summary, top_k=retrieve_n)
        logger.info(f"Retrieved {len(retrieved)} clauses from {n_trials} indexed trials")

        # 2. Group by trial
        trial_clause_map: dict[str, list[dict]] = defaultdict(list)
        for clause in retrieved:
            trial_clause_map[clause["nct_id"]].append(clause)

        # 3. For each candidate trial, run rule checks + compute score
        scored_trials = []
        for nct_id, clauses in trial_clause_map.items():
            trial = self.trials.get(nct_id)
            if not trial:
                continue

            # Rule verification on normalized constraints
            all_passed = []
            all_failed = []
            all_uncertain = []

            for clause in trial.get("clauses", []):
                constraints = clause.get("normalized") or normalize_clause(clause["text"])
                result = verify_constraints(
                    constraints, patient, is_inclusion=clause.get("is_inclusion", True)
                )
                all_passed.extend(result.passed)
                all_failed.extend(result.failed)
                all_uncertain.extend(result.uncertain)

            # Rule score
            if all_failed:
                rule_score = 0.0
            elif all_uncertain and not all_passed:
                rule_score = 0.5
            elif all_passed:
                rule_score = 1.0
            else:
                rule_score = 0.5

            # Semantic score from inclusion clause similarities
            inc_sims = [c["similarity"] for c in clauses if c.get("is_inclusion", True)]
            sem_score = compute_semantic_score(inc_sims)
            meta_score = compute_trial_meta_score(trial)

            final_score = score_trial(trial, rule_score, inc_sims)

            scored_trials.append({
                "trial": trial,
                "score": final_score,
                "rule_score": rule_score,
                "semantic_score": sem_score,
                "trial_meta_score": meta_score,
                "matched_clauses": [c for c in clauses if c.get("is_inclusion", True)],
                "violated_clauses": all_failed,
                "audit": {
                    "passed": all_passed,
                    "failed": all_failed,
                    "uncertain": all_uncertain,
                } if debug else None,
            })

        # Sort and take top-k
        scored_trials.sort(key=lambda x: x["score"], reverse=True)
        scored_trials = scored_trials[:top_k]

        # 4. Generate explanations
        matches = []
        for item in scored_trials:
            trial = item["trial"]
            confidence = (
                "high" if item["rule_score"] == 1.0 and item["semantic_score"] > 0.6
                else "low" if item["rule_score"] == 0.0
                else "medium"
            )

            # Get rationale (LLM or fallback)
            rationale = await self._generate_rationale(
                patient, trial, item["matched_clauses"], item["violated_clauses"], confidence
            )

            matches.append(TrialMatch(
                nct_id=trial["nct_id"],
                title=trial.get("title", ""),
                score=item["score"],
                rule_score=item["rule_score"],
                semantic_score=item["semantic_score"],
                trial_meta_score=item["trial_meta_score"],
                matched_clauses=[
                    MatchedClause(
                        clause_id=c.get("clause_id", ""),
                        text=c.get("text", ""),
                        type=c.get("type", "other"),
                        similarity=c.get("similarity"),
                    )
                    for c in item["matched_clauses"][:5]
                ],
                violated_clauses=[
                    MatchedClause(
                        clause_id=v.get("clause_id", v.get("type", "unknown")),
                        text=str(v.get("reason", v.get("text", ""))),
                        type=v.get("type", "other"),
                    )
                    for v in item["violated_clauses"]
                ],
                rationale=rationale,
                confidence=confidence,
                evidence=[
                    EvidenceCitation(
                        snippet=c.get("text", "")[:120],
                        url=trial.get("url"),
                    )
                    for c in item["matched_clauses"][:3]
                ],
            ))

        return MatchResponse(
            patient_id=patient.patient_id,
            matches=matches,
            audit={"retrieved_count": len(retrieved)} if debug else None,
        )

    async def _generate_rationale(
        self,
        patient: PatientInput,
        trial: dict,
        matched: list[dict],
        violated: list[dict],
        confidence: str,
    ) -> str:
        if self.llm_client:
            try:
                from llm.explainer import generate_explanation
                return await generate_explanation(
                    self.llm_client, patient, trial, matched, violated, confidence
                )
            except Exception as e:
                logger.warning(f"LLM explanation failed: {e}")

        # Fallback: rule-based rationale
        mc = [c.get("text", "")[:80] for c in matched[:3]]
        vc = [v.get("reason", v.get("text", ""))[:80] for v in violated[:3]]
        parts = [f"Patient (age {patient.age}, {patient.sex}) retrieved as candidate for '{trial.get('title', '')}'."]
        if mc:
            parts.append(f"Matching criteria include: {'; '.join(mc)}.")
        if vc:
            parts.append(f"Potential concerns: {'; '.join(vc)}.")
        parts.append(f"Confidence: {confidence}. Clinician review required.")
        return " ".join(parts)