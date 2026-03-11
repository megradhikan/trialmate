# TrialMate — Clinical Trial Matching Agent: Design Document

> ⚠️ **PROTOTYPE ONLY** — Not for clinical use. Uses synthetic data only. Do not process real PHI.

---

## 1. Architecture Overview

```
Patient JSON ──► Ingest & Normalize ──► Patient Embedding
                                              │
ClinicalTrials.gov XML/JSON                   ▼
    │                                  Vector Retrieval (FAISS/Chroma)
    ▼                                         │
Trial Ingestion ──► Clause Splitting ──►  Clause Embeddings
    │                     │                   │
    ▼                     ▼                   ▼
Normalized JSON    Clause Index ◄──────  Top-K Clauses
                                              │
                              ┌───────────────┤
                              ▼               ▼
                       Rule Verifier    LLM Normalizer
                       (age,sex,labs)   (ambiguous clauses)
                              │               │
                              └───────┬───────┘
                                      ▼
                                Scoring Module
                                  w_rule * rule_score
                                + w_sem  * semantic_score
                                + w_trial* trial_meta_score
                                      │
                                      ▼
                              Explanation Generator (LLM)
                                      │
                                      ▼
                              FastAPI Response (ranked matches)
```

---

## 2. Data Flow

1. **Trial Ingestion** (`ingestion/trial_ingestor.py`): Downloads or reads ClinicalTrials.gov JSON/XML, extracts `nct_id`, title, phase, status, eligibility text.
2. **Clause Splitting** (`ingestion/clause_splitter.py`): Uses LLM (or regex fallback) to split eligibility free-text into atomic clauses. Each clause is assigned a `clause_id` and `type_hint`.
3. **Clause Normalization** (`matching/clause_normalizer.py`): Deterministic regex parsers extract structured constraints (age range, sex, lab thresholds). For failures, LLM fallback.
4. **Embedding & Indexing** (`matching/embedder.py`): SentenceTransformers or OpenAI embeds clause text → FAISS index.
5. **Patient Ingestion** (`api/patient_schema.py` + `ingestion/patient_ingestor.py`): Validates and normalizes FHIR-like patient JSON.
6. **Retrieval** (`matching/retriever.py`): Patient summary embedded → FAISS ANN search → top-K clauses.
7. **Rule Verification** (`matching/rule_verifier.py`): Deterministic check of parsed constraints against patient values.
8. **Scoring** (`matching/scorer.py`): Combines rule_score + semantic_score + trial_meta_score.
9. **Explanation** (`llm/explainer.py`): LLM generates rationale citing clause IDs.
10. **API Layer** (`api/main.py`): FastAPI endpoints, response caching, audit logging.

---

## 3. Components & Responsibilities

| Component | File | Responsibility |
|-----------|------|----------------|
| Trial Ingestor | `ingestion/trial_ingestor.py` | Download, parse, save trials |
| Clause Splitter | `ingestion/clause_splitter.py` | Split eligibility text → atoms |
| Clause Normalizer | `matching/clause_normalizer.py` | Regex + LLM → structured constraints |
| Rule Verifier | `matching/rule_verifier.py` | Deterministic pass/fail per constraint |
| Embedder | `matching/embedder.py` | Embed text, build/load FAISS index |
| Retriever | `matching/retriever.py` | ANN search, return top-K clauses |
| Scorer | `matching/scorer.py` | Combine scores → 0–1 match score |
| LLM Client | `llm/client.py` | OpenAI/local model wrapper, caching |
| Explainer | `llm/explainer.py` | Rationale generation prompt + parsing |
| FastAPI App | `api/main.py` | REST endpoints, session management |
| Synthetic Gen | `scripts/generate_patients.py` | Synthea-like FHIR patient generator |
| Evaluator | `evaluation/evaluate.py` | Precision@k, Recall@k, F1, ablation |

---

## 4. Models Used

### Embeddings
- **Default (local)**: `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim)
- **Better quality**: `sentence-transformers/all-mpnet-base-v2` (768-dim)
- **OpenAI option**: `text-embedding-3-small` (cost-efficient) or `text-embedding-3-large`

### LLM for Clause Splitting / Normalization / Explanation
- **Default**: `claude-sonnet-4-20250514` via Anthropic API
- **OpenAI option**: `gpt-4o-mini` (fast, cheap) or `gpt-4o` (best)
- **Local fallback**: `mistral-7b-instruct` via Ollama

### Vector Store
- **Default**: FAISS (local, no server needed)
- **Alternative**: ChromaDB (persistent, with metadata filtering)

---

## 5. Scoring Formula

```
score = w_rule * rule_score + w_sem * semantic_score + w_trial * trial_meta_score
```

Default weights: `w_rule=0.6, w_sem=0.3, w_trial=0.1`

- **rule_score**: 1.0 = all deterministic checks pass; 0.0 = any hard exclusion; 0.5 = uncertain
- **semantic_score**: mean cosine similarity (patient embedding vs inclusion clauses), normalized 0–1
- **trial_meta_score**: 1.0=Recruiting, 0.7=Not Yet Recruiting, 0.5=Active Not Recruiting, 0.0=Terminated/Suspended

---

## 6. Key Tradeoffs

| Decision | Choice | Rationale |
|----------|--------|-----------|
| FAISS vs Chroma | FAISS default | Zero-server, fast ANN; Chroma for metadata filtering |
| LLM for splitting | LLM + regex fallback | LLM handles complex syntax; regex is faster and deterministic |
| OpenAI vs local | Both options provided | OpenAI = best quality; local = privacy, no API cost |
| Clause granularity | Atomic (1 constraint per clause) | Enables precise rule checks and explanations |
| PHI handling | Synthetic data only | No PHI ever stored; session data deleted post-request |

---

## 7. Security & Ethics

- All default data is **synthetic** (generated via script, not Synthea HIPAA data)
- No PHI stored on disk; in-memory patient objects are deleted after each request
- LLM prompts are logged (without patient data) for audit
- System labeled PROTOTYPE prominently in all interfaces
- If real patient data is uploaded, UI shows a warning and auto-deletes after session
- No production deployment without IRB/HIPAA compliance review

---

## 8. Assumptions

1. ClinicalTrials.gov data is accessed via public API (no auth required)
2. Patient records use simplified FHIR-like JSON (not full R4)
3. Labs use standard test names (mapped to LOINC in future work)
4. Prototype targets ~1000 trials (scalable to full dataset with same architecture)
5. Latency budget: vector retrieval <500ms; LLM explanation <3s with caching
