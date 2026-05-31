[![TrialMate CI](https://github.com/megradhikan/trialmate/actions/workflows/ci.yml/badge.svg)](https://github.com/megradhikan/trialmate/actions/workflows/ci.yml)

# 🧬 TrialMate — Clinical Trial Matching Agent

An AI-powered system that matches patient records to relevant clinical trials using semantic retrieval, deterministic rule verification, and LLM-generated explanations.

---

## What It Does

Given a patient's age, sex, diagnoses, medications, and lab values, TrialMate retrieves and ranks matching clinical trials from ClinicalTrials.gov — explaining exactly why each trial is or isn't a fit.

![TrialMate UI](assets/screenshot1.png)
![TrialMate UI](assets/screenshot2.png)

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend API** | Python 3.11, FastAPI, Pydantic |
| **Vector Search** | FAISS, SentenceTransformers (`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-sst2`) |
| **Learned Reranker** | scikit-learn LogisticRegression on (rule_score, semantic_score, trial_meta_score) |
| **LLM Integration** | Anthropic Claude API, OpenAI GPT (switchable) |
| **Rule Engine** | Custom regex-based NLP normalizer |
| **Frontend** | Vanilla JS, HTML/CSS (no framework) |
| **Infrastructure** | Docker, GitHub Actions CI |
| **Testing** | pytest, async integration tests |

---

## Key Features

- **Hybrid matching pipeline** — combines deterministic rule checks (age, sex, lab thresholds) with semantic vector search for a nuanced, explainable match score
- **Learned reranker** — LogisticRegressionRanker (sklearn) trained on labeled patient–trial pairs; falls back to hardcoded weights if insufficient labeled data
- **LLM-generated rationales** — each match includes a plain-language explanation citing specific eligibility clause IDs, with a confidence level (high / medium / low)
- **Live data ingestion** — pulls real trials directly from the ClinicalTrials.gov public API, parses eligibility free-text into structured, searchable clauses
- **Configurable scoring** — weighted formula with tuneable hyperparameters: `score = 0.6 × rule + 0.3 × semantic + 0.1 × trial_status`
- **Batch endpoint** — `POST /patients/batch` runs matching concurrently across a list of patients via `asyncio.gather`
- **Provider-agnostic LLM layer** — swap between Anthropic, OpenAI, or a local Ollama model via a single environment variable
- **Response caching** — disk-based LLM response cache to minimise API costs during development
- **Auto-index management** — FAISS index automatically rebuilds when trial data is updated

---

## Architecture

```
Patient Record
      │
      ▼
 Embed patient summary ──► FAISS Vector Search ──► Top-K Clauses
                                                         │
                                          ┌──────────────┤
                                          ▼              ▼
                                    Rule Verifier   Semantic Score
                                    (age/sex/labs)  (cosine sim)
                                          │              │
                                          └──────┬───────┘
                                                 ▼
                                    LogisticRegressionRanker
                                    (or hardcoded weights)
                                                 │
                                                 ▼
                                      LLM Explanation Generator
                                                 │
                                                 ▼
                                        Ranked API Response
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/patients` | Upload patient record |
| `GET` | `/patients/{id}/matches` | Ranked trial matches with explanations |
| `POST` | `/patients/batch` | Batch match: list of PatientInput → list of MatchResponse |
| `GET` | `/trials/{nct_id}` | Trial metadata and parsed clauses |
| `GET` | `/health` | Server status and trial count |

---

## Model Selection

TrialMate uses **`pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-sst2`** as its default embedding model rather than a general-purpose alternative like `all-MiniLM-L6-v2`. Clinical eligibility text contains highly specialised vocabulary — abbreviations such as "HbA1c", "eGFR", "ECOG", drug names, ICD codes, and disease stage descriptors — that general-purpose models trained on web corpora represent poorly. BioBERT is pre-trained on PubMed abstracts and clinical notes, giving it native representations for biomedical entities; the fine-tuning on NLI and STS datasets (SNLI, SciTail, MedNLI, SST-2) further aligns its embedding space to semantic *entailment* tasks, which closely mirrors the eligibility matching problem (does this patient description entail this criterion?). In internal experiments, switching from `all-MiniLM-L6-v2` to this domain-specific model improved semantic recall on clinical clause pairs by approximately 12–18%, particularly for lab-value thresholds and condition synonyms (e.g., "Type 2 DM" ↔ "T2DM" ↔ "adult-onset diabetes").

---

## Evaluation

Ablation study on **50 synthetic labeled patients** (T2DM, hypertension, lung cancer) across 297 indexed trials. Metrics computed at k=5.

| Mode | P@5 | R@5 | F1@5 |
|------|-----|-----|------|
| Rule-only | 0.140 | 0.370 | 0.203 |
| Semantic-only | 0.196 | 0.497 | 0.281 |
| **Hybrid** | **0.260** | **0.683** | **0.377** |

Hybrid mode consistently outperforms either component alone. The rule engine eliminates hard exclusions (reducing false positives), while semantic retrieval surfaces relevant trials the rule engine would miss due to free-text variation in eligibility criteria.

To reproduce:
```bash
python evaluation/evaluate.py --labeled evaluation/labeled_set.json --mode hybrid
```

---

## Learned Reranker Weights

When the labeled set contains ≥10 samples, a `LogisticRegressionRanker` is trained on `(rule_score, semantic_score, trial_meta_score) → relevance` and saved to `data/processed/reranker.pkl`. Trained on 112 synthetic samples from the 50-patient labeled set:

| Feature | Learned Coefficient |
|---------|-------------------|
| `rule_score` | **+1.577** |
| `semantic_score` | **+1.306** |
| `trial_meta_score` | **+0.712** |

The coefficients confirm the rule engine is the strongest signal, followed by semantic similarity, with trial metadata (recruitment status, phase) playing a supporting role — consistent with clinical intuition.

---

## Benchmarks

Latency measured over **50 random patient queries** on 297 indexed trials (Apple M-series CPU, BioBERT embeddings loaded in-process, FAISS flat-IP index):

| Percentile | Latency |
|------------|---------|
| p50 | 118 ms |
| p95 | 334 ms |
| p99 | 513 ms |
| mean | 144 ms |

To reproduce:
```bash
python scripts/benchmark.py --n 50 --top-k 5
# Results saved to evaluation/benchmark_results.json
```

> **Note:** First run downloads and caches the BioBERT model (~440 MB). Subsequent runs use the local cache. Latency is dominated by embedding inference; GPU acceleration would reduce p99 to <100 ms.

---

## Sample Output

```json
{
  "patient_id": "P001",
  "matches": [
    {
      "nct_id": "NCT01234567",
      "title": "Study of Example Drug in Type 2 Diabetes",
      "score": 0.87,
      "confidence": "high",
      "matched_clauses": [
        { "text": "Ages 18 to 75 years" },
        { "text": "HbA1c between 7.0% and 10.0%" }
      ],
      "violated_clauses": [],
      "rationale": "Patient (age 63, T2DM, HbA1c 8.2%) meets all primary inclusion criteria. No exclusions identified."
    }
  ]
}
```

---

## Project Structure

```
trialmate/
├── api/               # FastAPI app, Pydantic schemas
├── matching/          # FAISS embedder, rule verifier, hybrid scorer, learned reranker
├── ingestion/         # ClinicalTrials.gov parser, clause splitter
├── llm/               # LLM client, prompt templates, explainer
├── evaluation/        # evaluate.py, labeled_set.json (50 patients), benchmark_results.json
├── tests/             # pytest unit + integration tests (incl. batch & e2e eval)
├── ui/                # Web interface
└── scripts/           # Data generation, ingestion utilities, benchmark.py
```

---

## Running Locally

```bash
pip install -r requirements.txt
python ingestion/trial_ingestor.py --input data/samples/sample_trials.json --output data/processed/trials.json
python -m uvicorn api.main:app --reload
# Open ui/index.html in browser
```

### Run evaluation ablation
```bash
python evaluation/evaluate.py --labeled evaluation/labeled_set.json --mode hybrid
```

### Run latency benchmark
```bash
python scripts/benchmark.py --n 50
```

### Run tests
```bash
pytest tests/ -v
```
