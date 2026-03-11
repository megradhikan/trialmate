# TrialMate — Security & Ethics Documentation

> This document must be reviewed before any deployment or use of TrialMate.

---

## 1. PHI Policy (Protected Health Information)

**TrialMate is designed to operate exclusively on synthetic data. It must NEVER process real patient data.**

### Rules
- Default dataset is generated synthetically via `scripts/generate_patients.py`
- If real patient data is used (e.g., de-identified MIMIC-III), operators must:
  - Obtain IRB approval
  - Complete HIPAA Business Associate Agreement with all cloud providers
  - Never store data beyond the session duration
  - Purge all logs containing patient fields
- Patient records are stored **in-memory only** during a session
- On server shutdown, all in-memory patient data is automatically cleared
- No patient data is written to disk, databases, or log files

### Implementation Safeguards
- `api/main.py` uses a dict `_patient_store` that lives only in process memory
- Logs are filtered to exclude patient fields (only metadata: patient_id, age range, not identifiers)
- LLM prompts include patient summaries but are never stored in plain form unless in debug mode with user consent

---

## 2. System Classification

- **Classification**: Research prototype
- **Regulatory status**: NOT FDA-cleared, NOT CE-marked
- **Intended use**: Demonstrating feasibility of AI-assisted trial matching with synthetic data only
- **Prohibited uses**: Clinical decision-making, actual patient screening, production clinical workflows

---

## 3. AI Model Transparency

- LLM-generated rationales are clearly labeled as AI-generated
- All responses include a `confidence` field (low/medium/high) with explicit uncertainty
- Violated clauses and uncertain items are surfaced to the clinician
- The system does NOT make binary eligible/ineligible decisions — it produces ranked candidates for human review

---

## 4. Data Sources

| Source | Status | Notes |
|--------|--------|-------|
| Synthetic patients (generate_patients.py) | ✅ Safe | Random generation, no real data |
| ClinicalTrials.gov public API | ✅ Safe | Public domain, no PHI |
| Synthea-generated FHIR data | ✅ Safe | Synthetic, open-source |
| MIMIC-III/IV | ⚠️ Requires approval | PhysioNet credentialing + IRB required |
| Real EHR data | ❌ Not permitted | Never use without full compliance review |

---

## 5. Security Recommendations for Deployment

If this prototype is ever deployed beyond a local laptop:

1. **Authentication**: Add API key auth or OAuth2 to all endpoints
2. **HTTPS**: Use TLS (nginx + certbot) — never HTTP for patient data
3. **Rate limiting**: Add rate limits to prevent abuse
4. **Audit logging**: Log all API calls (without PHI) with timestamps
5. **Secret management**: Never hardcode API keys; use environment variables or a secrets manager
6. **Network isolation**: Run in a VPC with no public internet exposure if using real data

---

## 6. Bias & Fairness Considerations

- Embedding models may have biases toward certain medical terminology (English-centric)
- Clinical trial eligibility criteria historically underrepresent certain demographic groups
- TrialMate's rankings reflect the biases present in ClinicalTrials.gov data
- All results require clinician review; the system is a decision-support tool only

---

## 7. Incident Response

If real patient data is accidentally processed:
1. Immediately stop the server: `kill $(lsof -t -i:8000)`
2. Delete `.llm_cache/` directory
3. Notify data protection officer within 72 hours (GDPR) or as required by applicable law
4. Document the incident

---

## 8. Contact

For questions about appropriate use, contact the project maintainer before deploying.
