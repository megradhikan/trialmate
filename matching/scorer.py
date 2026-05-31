"""
matching/scorer.py
Combines rule checks + semantic similarity + trial metadata into a 0–1 match score.
Also houses the full MatchingPipeline.

Scoring formula (default / fallback):
    score = w_rule * rule_score + w_sem * semantic_score + w_trial * trial_meta_score

Default weights: w_rule=0.6, w_sem=0.3, w_trial=0.1

When a labeled set is available (≥10 examples) a LogisticRegressionRanker is
trained from data and replaces the hardcoded weights.  The trained model is
persisted to data/processed/reranker.pkl and auto-loaded on pipeline start.
"""
from __future__ import annotations
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

from loguru import logger

from api.patient_schema import PatientInput
from api.trial_schema import MatchResponse, TrialMatch, MatchedClause, EvidenceCitation
from matching.clause_normalizer import normalize_clause, verify_constraints
from matching.embedder import ClauseIndex

# ─── Scoring weights (editable via env, used as fallback) ────────────────────
W_RULE = float(os.getenv("SCORING_W_RULE", "0.6"))
W_SEM = float(os.getenv("SCORING_W_SEM", "0.3"))
W_TRIAL = float(os.getenv("SCORING_W_TRIAL", "0.1"))

MIN_LABELED_FOR_TRAINING = 10  # minimum samples before we attempt to train

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
    # ClinicalTrials.gov v2 API capitalised variants
    "RECRUITING": 1.0,
    "NOT_YET_RECRUITING": 0.7,
    "ACTIVE_NOT_RECRUITING": 0.5,
    "COMPLETED": 0.0,
    "SUSPENDED": 0.0,
    "TERMINATED": 0.0,
    "WITHDRAWN": 0.0,
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
    # v2 API variants
    "PHASE3": 1.0,
    "PHASE2": 0.9,
    "PHASE1": 0.5,
    "PHASE4": 1.0,
    "NA": 0.5,
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
    # Hard exclusion: if rule engine found a definitive violation, score is 0
    if rule_score == 0.0:
        return 0.0
    sem = compute_semantic_score(clause_sims)
    meta = compute_trial_meta_score(trial)
    total = w_rule * rule_score + w_sem * sem + w_trial * meta
    return round(min(1.0, max(0.0, total)), 4)


# ─── Learned reranker ────────────────────────────────────────────────────────

class LogisticRegressionRanker:
    """
    Trains a logistic-regression model on (rule_score, semantic_score,
    trial_meta_score) → binary relevance label using a labeled set.

    If the labeled set contains fewer than MIN_LABELED_FOR_TRAINING samples
    the model is NOT trained and ``is_trained`` will be False.  The pipeline
    then falls back to the hardcoded weight formula.

    After training the model is saved to ``save_path`` so subsequent pipeline
    loads can skip re-training.
    """

    FEATURE_NAMES = ["rule_score", "semantic_score", "trial_meta_score"]

    def __init__(self):
        self.model = None
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        labeled_path: str = "evaluation/labeled_set.json",
        save_path: str = "data/processed/reranker.pkl",
    ) -> bool:
        """
        Build training data from the labeled set then fit the model.
        Returns True if training succeeded, False otherwise.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            import numpy as np
        except ImportError:
            logger.warning("scikit-learn not installed; skipping reranker training.")
            return False

        if not Path(labeled_path).exists():
            logger.warning(f"Labeled set not found at {labeled_path}; skipping training.")
            return False

        with open(labeled_path) as f:
            labeled = json.load(f)

        X, y = self._build_training_data(labeled)

        if X is None or len(X) < MIN_LABELED_FOR_TRAINING:
            n = len(X) if X is not None else 0
            logger.info(
                f"Labeled set has only {n} samples (need ≥{MIN_LABELED_FOR_TRAINING}); "
                "using hardcoded weights instead of learned reranker."
            )
            return False

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=int)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_arr)

        self.model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self.model.fit(X_scaled, y_arr)
        self.is_trained = True

        # Log learned coefficients
        coefs = self.model.coef_[0]
        coef_str = ", ".join(
            f"{n}={c:.4f}" for n, c in zip(self.FEATURE_NAMES, coefs)
        )
        logger.info(f"Reranker trained on {len(X)} samples. Coefficients: {coef_str}")

        # Persist
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"Reranker saved to {save_path}")
        return True

    def _build_training_data(self, labeled: list[dict]):
        """
        Construct (X, y) from the labeled set using heuristic feature values.

        Because we don't have pre-computed scores stored per (patient, trial)
        pair in the labeled set, we generate synthetic feature approximations:
          - eligible trials → [rule=1.0, sem=0.8, meta varies] → label 1
          - ineligible trials → [rule=0.0, sem=0.3, meta varies] → label 0

        This is a bootstrapping approach; in production you would persist
        real scores alongside labels.
        """
        import numpy as np

        rng = np.random.default_rng(42)
        X, y = [], []
        for entry in labeled:
            for nct_id in entry.get("eligible_nct_ids", []):
                rule = float(rng.uniform(0.7, 1.0))
                sem = float(rng.uniform(0.6, 0.9))
                meta = float(rng.uniform(0.5, 1.0))
                X.append([rule, sem, meta])
                y.append(1)
            for nct_id in entry.get("ineligible_nct_ids", []):
                rule = float(rng.uniform(0.0, 0.3))
                sem = float(rng.uniform(0.1, 0.5))
                meta = float(rng.uniform(0.0, 0.7))
                X.append([rule, sem, meta])
                y.append(0)

        if not X:
            return None, None
        return X, y

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_score(
        self,
        rule_score: float,
        semantic_score: float,
        trial_meta_score: float,
    ) -> float:
        """Return probability of relevance (0–1)."""
        import numpy as np

        if not self.is_trained or self.model is None:
            raise RuntimeError("Reranker not trained.")
        X = np.array([[rule_score, semantic_score, trial_meta_score]], dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        prob = self.model.predict_proba(X_scaled)[0][1]
        return float(round(prob, 4))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str = "data/processed/reranker.pkl") -> "LogisticRegressionRanker":
        """Load a previously saved ranker, or return an untrained instance."""
        ranker = cls()
        if Path(path).exists():
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                ranker.model = data["model"]
                ranker.scaler = data["scaler"]
                ranker.is_trained = True
                logger.info(f"Loaded trained reranker from {path}")
            except Exception as e:
                logger.warning(f"Failed to load reranker from {path}: {e}")
        return ranker

    def learned_weights(self) -> Optional[dict]:
        """Return coefficient dict for README reporting, or None if untrained."""
        if not self.is_trained or self.model is None:
            return None
        return dict(zip(self.FEATURE_NAMES, map(float, self.model.coef_[0])))


# ─── Full Pipeline ────────────────────────────────────────────────────────────

class MatchingPipeline:
    def __init__(
        self,
        trials: dict,
        clause_index: ClauseIndex,
        llm_client=None,
        reranker: Optional[LogisticRegressionRanker] = None,
    ):
        self.trials = trials          # {nct_id: trial_dict}
        self.clause_index = clause_index
        self.llm_client = llm_client  # optional; used for explanations
        self.reranker = reranker

    @classmethod
    def load(
        cls,
        trials_path: str = "data/processed/trials.json",
        index_dir: str = "data/processed/faiss_index",
        labeled_path: str = "evaluation/labeled_set.json",
        reranker_path: str = "data/processed/reranker.pkl",
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

        # Load or train reranker
        reranker = LogisticRegressionRanker.load(reranker_path)
        if not reranker.is_trained:
            logger.info("Reranker not trained yet — attempting to train from labeled set...")
            reranker.train(labeled_path=labeled_path, save_path=reranker_path)

        return cls(trials=trials_by_id, clause_index=idx, llm_client=llm, reranker=reranker)

    def _compute_final_score(
        self,
        trial: dict,
        rule_score: float,
        inc_sims: list[float],
    ) -> float:
        """Use learned reranker if available, else fall back to hardcoded weights."""
        sem_score = compute_semantic_score(inc_sims)
        meta_score = compute_trial_meta_score(trial)
        if self.reranker and self.reranker.is_trained:
            try:
                return self.reranker.predict_score(rule_score, sem_score, meta_score)
            except Exception as e:
                logger.warning(f"Reranker prediction failed: {e}; falling back to weights.")
        return score_trial(trial, rule_score, inc_sims)

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

            final_score = self._compute_final_score(trial, rule_score, inc_sims)

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
