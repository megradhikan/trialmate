"""
tests/test_clause_normalizer.py
Unit tests for clause normalizer and rule verifier.
Run: pytest tests/ -v
"""
from __future__ import annotations
import pytest
import sys
sys.path.insert(0, ".")

from matching.clause_normalizer import normalize_clause, verify_constraints
from api.patient_schema import PatientInput, PatientLabs


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_patient(**kwargs) -> PatientInput:
    defaults = dict(
        patient_id="TEST001",
        age=55,
        sex="male",
        conditions=["type 2 diabetes"],
        medications=["metformin"],
        labs=PatientLabs(hemoglobin_a1c=8.2, creatinine=0.9, egfr=75.0),
    )
    defaults.update(kwargs)
    return PatientInput(**defaults)


# ─── Clause Normalizer Tests ─────────────────────────────────────────────────

class TestClauseNormalizer:

    def test_age_range(self):
        c = normalize_clause("Ages 18 to 75 years")
        assert any(x["type"] == "age" for x in c)
        age = next(x for x in c if x["type"] == "age")
        assert age["min"] == 18
        assert age["max"] == 75

    def test_age_min_only(self):
        c = normalize_clause("Patients must be at least 21 years old")
        assert any(x["type"] == "age" and x.get("min") == 21 for x in c)

    def test_age_max_only(self):
        c = normalize_clause("Age ≤ 65 years")
        assert any(x["type"] == "age" and x.get("max") == 65 for x in c)

    def test_hba1c_between(self):
        c = normalize_clause("HbA1c between 7.0% and 10.0%")
        lab_cs = [x for x in c if x["type"] == "lab"]
        assert len(lab_cs) == 2
        assert any(x["op"] == ">=" and x["value"] == 7.0 for x in lab_cs)
        assert any(x["op"] == "<=" and x["value"] == 10.0 for x in lab_cs)

    def test_creatinine_threshold(self):
        c = normalize_clause("Serum creatinine ≤ 1.5 mg/dL")
        lab_cs = [x for x in c if x["type"] == "lab"]
        assert any(x["test"] == "creatinine" and x["op"] == "<=" for x in lab_cs)

    def test_sex_female_only(self):
        c = normalize_clause("Female patients only")
        sex_cs = [x for x in c if x["type"] == "sex"]
        assert sex_cs and "female" in sex_cs[0]["allowed"]

    def test_sex_both(self):
        c = normalize_clause("Male or female patients aged 18 and older")
        # Should not restrict to single sex
        sex_cs = [x for x in c if x["type"] == "sex"]
        if sex_cs:
            assert len(sex_cs[0]["allowed"]) > 1

    def test_pregnancy_exclusion(self):
        c = normalize_clause("Not pregnant or breastfeeding")
        excl = [x for x in c if x["type"] == "exclusion"]
        assert excl

    def test_egfr_threshold(self):
        c = normalize_clause("eGFR >= 30 mL/min/1.73m²")
        lab_cs = [x for x in c if x["type"] == "lab" and x.get("test") == "egfr"]
        assert lab_cs

    def test_empty_clause(self):
        c = normalize_clause("Patients must be willing and able to provide consent")
        # Consent clause has no deterministic constraint
        assert isinstance(c, list)  # Returns empty list or uncertain items

    def test_complex_clause_no_crash(self):
        c = normalize_clause(
            "Prior treatment with any GLP-1 receptor agonist within the last 6 months "
            "or current enrollment in another interventional clinical trial"
        )
        assert isinstance(c, list)


# ─── Rule Verifier Tests ─────────────────────────────────────────────────────

class TestRuleVerifier:

    def test_age_pass(self):
        patient = make_patient(age=55)
        constraints = [{"type": "age", "min": 18, "max": 75}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert result.rule_score == 1.0
        assert not result.has_hard_exclusion

    def test_age_fail_too_young(self):
        patient = make_patient(age=15)
        constraints = [{"type": "age", "min": 18, "max": 75}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert result.has_hard_exclusion
        assert result.rule_score == 0.0

    def test_age_fail_too_old(self):
        patient = make_patient(age=90)
        constraints = [{"type": "age", "min": 18, "max": 75}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert result.has_hard_exclusion

    def test_sex_pass(self):
        patient = make_patient(sex="male")
        constraints = [{"type": "sex", "allowed": ["male"]}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert not result.has_hard_exclusion

    def test_sex_fail(self):
        patient = make_patient(sex="female")
        constraints = [{"type": "sex", "allowed": ["male"]}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert result.has_hard_exclusion

    def test_lab_pass(self):
        patient = make_patient()  # hba1c=8.2
        constraints = [{"type": "lab", "test": "hemoglobin_a1c", "op": ">=", "value": 7.0},
                       {"type": "lab", "test": "hemoglobin_a1c", "op": "<=", "value": 10.0}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert not result.has_hard_exclusion

    def test_lab_fail_too_high(self):
        patient = make_patient()  # hba1c=8.2
        constraints = [{"type": "lab", "test": "hemoglobin_a1c", "op": "<=", "value": 7.0}]
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert result.has_hard_exclusion

    def test_lab_missing_uncertain(self):
        patient = make_patient()
        constraints = [{"type": "lab", "test": "wbc", "op": ">=", "value": 4.0}]
        # WBC not in patient labs
        result = verify_constraints(constraints, patient, is_inclusion=True)
        assert result.uncertain  # uncertain, not failed

    def test_exclusion_clause_uncertain(self):
        patient = make_patient()
        constraints = [{"type": "exclusion", "text": "History of severe liver disease"}]
        result = verify_constraints(constraints, patient, is_inclusion=False)
        assert result.uncertain

    def test_no_constraints_returns_medium(self):
        patient = make_patient()
        result = verify_constraints([], patient, is_inclusion=True)
        assert result.rule_score == 0.5


# ─── Scorer Tests ─────────────────────────────────────────────────────────────

class TestScorer:

    def test_hard_exclusion_zero_score(self):
        from matching.scorer import score_trial
        trial = {"status": "Recruiting", "phase": "Phase 3"}
        score = score_trial(trial, rule_score=0.0, clause_sims=[0.9, 0.8])
        assert score == 0.0

    def test_high_match_high_score(self):
        from matching.scorer import score_trial
        trial = {"status": "Recruiting", "phase": "Phase 3"}
        score = score_trial(trial, rule_score=1.0, clause_sims=[0.85, 0.90, 0.82])
        assert score > 0.8

    def test_terminated_lowers_score(self):
        from matching.scorer import score_trial
        trial_active = {"status": "Recruiting", "phase": "Phase 3"}
        trial_term = {"status": "Terminated", "phase": "Phase 3"}
        s1 = score_trial(trial_active, rule_score=1.0, clause_sims=[0.8])
        s2 = score_trial(trial_term, rule_score=1.0, clause_sims=[0.8])
        assert s1 > s2

    def test_score_in_range(self):
        from matching.scorer import score_trial
        trial = {"status": "Unknown", "phase": "N/A"}
        score = score_trial(trial, rule_score=0.5, clause_sims=[0.5])
        assert 0.0 <= score <= 1.0
