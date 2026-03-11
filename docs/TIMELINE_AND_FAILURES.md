# TrialMate — Development Timeline (GitHub Issues Format)

## Week 0: Repository Setup & Data Foundation
**Issue #1 — Repo initialization and data download**
- [ ] Create GitHub repo, add README with ⚠️ prototype warning
- [ ] Add `.gitignore`, `requirements.txt`, `Dockerfile`, `.env.example`
- [ ] Run `scripts/generate_patients.py --count 200 --seed 42`
- [ ] Download sample ClinicalTrials.gov data via `scripts/download_trials.py`
- [ ] Commit `data/samples/` with at least 20 sample trials and 10 patients
- **Acceptance**: `python scripts/generate_patients.py` exits 0, outputs valid JSON

## Week 1: Trial Ingestion & Clause Splitting
**Issue #2 — Trial ingestor**
- [ ] Implement `ingestion/trial_ingestor.py` with ClinicalTrials.gov v2 API parsing
- [ ] Output normalized trial JSON with `nct_id`, `title`, `phase`, `status`, `clauses`
- [ ] Test with 50+ real trials
- **Acceptance**: `python ingestion/trial_ingestor.py --input data/raw/...` outputs valid JSON, 0 parse errors on sample set

**Issue #3 — Clause splitter**
- [ ] Implement regex-based clause splitter
- [ ] Implement LLM-assisted splitter with CLAUSE_SPLITTER_TEMPLATE
- [ ] Unit test with 10 messy eligibility texts
- **Acceptance**: splitter handles inclusion/exclusion sections, produces clause_ids, passes 8/10 test cases

## Week 2: Patient Ingestion & API Skeleton
**Issue #4 — Patient schemas and API**
- [ ] Implement Pydantic models in `api/patient_schema.py`, `api/trial_schema.py`
- [ ] Implement `POST /patients` and `GET /health` endpoints
- [ ] Add `to_summary_text()` method to PatientInput
- **Acceptance**: `pytest tests/test_api.py::TestPatientUpload` passes all 4 tests

## Week 3: Embeddings + Vector Index
**Issue #5 — Embedding and FAISS index**
- [ ] Implement `matching/embedder.py` with SentenceTransformers and OpenAI options
- [ ] Build FAISS index for 200+ trial clauses
- [ ] Implement basic `search()` returning top-K clause dicts
- **Acceptance**: Index builds in <60s on CPU, search returns results in <500ms

## Week 4: Rule Verifier + Unit Tests
**Issue #6 — Clause normalizer and rule verifier**
- [ ] Implement `matching/clause_normalizer.py` with regex patterns for age/sex/lab
- [ ] Implement `verify_constraints()` with pass/fail/uncertain output
- [ ] Write 20+ unit tests for normalizer
- [ ] Write 10+ unit tests for rule verifier
- **Acceptance**: `pytest tests/test_clause_normalizer.py` passes all tests, including hard exclusion → score 0.0

## Week 5: LLM Prompts & Normalization Module
**Issue #7 — LLM client and prompts**
- [ ] Implement `llm/client.py` with Anthropic + OpenAI + local Ollama options
- [ ] Implement disk cache for LLM responses
- [ ] Finalize all 3 prompt templates in `llm/prompts.py`
- [ ] Implement `llm/explainer.py` with `generate_explanation()`, `split_clauses_with_llm()`
- **Acceptance**: LLM client returns valid JSON for all 3 prompt templates on test inputs; cache reduces repeat call latency to <10ms

## Week 6: Matching, Scoring, End-to-End
**Issue #8 — Scorer and full pipeline**
- [ ] Implement `matching/scorer.py` with `score_trial()` and `MatchingPipeline`
- [ ] Wire up `GET /patients/{id}/matches` endpoint
- [ ] End-to-end test: upload P001, get 5 matches
- **Acceptance**: End-to-end pipeline runs in <5s total (excluding LLM); returned scores are in [0,1]; hard exclusion results in score 0.0

## Week 7: Explanation Generator & UI MVP
**Issue #9 — Explanation generation**
- [ ] Wire LLM explanation into `_generate_rationale()` with fallback
- [ ] Verify rationale cites clause_ids
- [ ] Implement `ui/index.html` demo
- **Acceptance**: UI loads, uploads sample patient, displays match cards with scores, rationale, and clause tags

## Week 8: Labeled Eval Set + Metrics
**Issue #10 — Evaluation framework**
- [ ] Expand `evaluation/labeled_set.json` to 100+ patient-trial pairs
- [ ] Run `evaluation/evaluate.py` for all 200 synthetic patients
- [ ] Report Precision@5, Recall@5, F1
- **Acceptance**: Metrics script outputs valid JSON; hybrid mode F1 > 0.5 on labeled set

## Week 9: Weight Tuning & Ablation
**Issue #11 — Ablation and weight calibration**
- [ ] Implement `evaluation/ablation.py` for semantic-only, rule-only, hybrid modes
- [ ] Grid search over w_rule, w_sem, w_trial on validation set
- [ ] Report ablation table
- **Acceptance**: Hybrid mode outperforms semantic-only and rule-only on F1@5; best weights documented in config

## Week 10: Finalize Docs, Demo, CI
**Issue #12 — Documentation and CI**
- [ ] Finalize `docs/DESIGN.md`, `docs/SECURITY_ETHICS.md`, `docs/PROMPT_TEMPLATES.md`
- [ ] Ensure `pytest tests/` passes in CI (GitHub Actions)
- [ ] Add Docker instructions and test `docker build && docker run`
- [ ] Record or document a demo walkthrough
- **Acceptance**: CI green on main branch; Docker image builds successfully; README complete with all setup instructions

---

# Top 12 Failure Modes & Mitigations

| # | Failure Mode | Description | Mitigation |
|---|-------------|-------------|-----------|
| 1 | **Clause splitter misses sections** | Eligibility text has no "Inclusion/Exclusion" headers, so splitter lumps everything | Add heuristic fallback: if no headers found, attempt numbered-list splitting; flag for LLM re-split |
| 2 | **Lab name mismatch** | Patient record uses "A1C" but normalizer looks for "hemoglobin_a1c" | Build a synonym map `{a1c, hba1c, glycated hemoglobin} → hemoglobin_a1c`; apply at ingestion |
| 3 | **Age expressed non-numerically** | "Adult patients" or "over 18" not caught by regex | Add patterns for "adult", "pediatric", "elderly"; map to numeric ranges |
| 4 | **LLM returns malformed JSON** | LLM response is valid English but not parseable JSON | Wrap all LLM output parsers in try/except with fallback to empty constraints; log the failure |
| 5 | **FAISS empty results for rare conditions** | Rare diseases have no similar clauses in index → zero retrieval | Add BM25 keyword fallback retrieval as a safety net; always return at least top-1 by keyword match |
| 6 | **Scoring weights too rule-heavy** | Rule checks always uncertain (missing labs) → all scores ~0.5, no differentiation | If >50% of constraints are uncertain, increase w_sem weight dynamically for that patient |
| 7 | **Trial metadata stale** | Status "Recruiting" in index but trial actually closed | Add a weekly re-ingestion cron job; cache-bust status field; display "last updated" date in UI |
| 8 | **Exclusion clauses not evaluated** | All retrieved clauses are inclusion; exclusions not checked → false positives | Explicitly retrieve and evaluate exclusion clauses separately in `MatchingPipeline.run()` |
| 9 | **LLM hallucinated clause interpretation** | LLM says patient meets criterion but text says otherwise | Add confidence=low to all LLM-only normalizations; require clinician review tag; never use LLM alone for hard exclusions |
| 10 | **Embedding model not domain-adapted** | General-purpose embeddings cluster "creatinine" and "clearance" poorly | Fine-tune or use BioBERT / PubMedBERT embeddings; document swap instructions in README |
| 11 | **Duplicate trials in results** | Same trial appears under multiple clause matches | Deduplicate by nct_id before scoring; keep highest-scoring clause set per trial |
| 12 | **API timeout on large trial set** | 10,000+ trials × vector search → latency >5s | Use FAISS IVF index (approximate) instead of flat; set `nprobe` hyperparameter; add response caching by patient summary hash |
