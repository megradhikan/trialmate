"""
matching/clause_normalizer.py
Normalizes eligibility clause text into structured constraint objects.
Uses regex for common patterns; falls back to LLM for ambiguous cases.
"""
from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.patient_schema import PatientInput


# ─── Regex Patterns ─────────────────────────────────────────────────────────

_AGE_CTX_RE = re.compile(r'\bag(?:e|es|ed)\b|\byears?\s+(?:of\s+age|old)\b|\d+\s*(?:years?|yrs?)\s*(?:and|or)\s+(?:older|younger)\b',
    re.IGNORECASE,
)
_UNIT = r'(?:years?|yrs?|y\b)'
AGE_RANGE_RE = re.compile(
    r'\bag(?:e|es|ed)\b[^0-9]{0,20}?(\d+)\s*' + _UNIT + r'?\s*(?:to|-|–|—|and)\s*(\d+)'
    r'|between\s+(\d+)\s*' + _UNIT + r'?\s*(?:and|to|-)\s*(\d+)\s*' + _UNIT,
    re.IGNORECASE
)
AGE_MIN_RE = re.compile(
    r'(?:≥|>=|at\s+least|minimum\s+age\s+of)\s*(\d+)'
    r'|(?:older\s+than|over|above|greater\s+than)\s+(\d+)\s*' + _UNIT +
    r'|(\d+)\s*' + _UNIT + r'?\s*(?:of\s+age\s*)?(?:and|or)\s+(?:older|above|over|greater)'
    r'|(\d+)\s*\+',
    re.IGNORECASE
)
AGE_MAX_RE = re.compile(
    r'(?:≤|<=|no\s+more\s+than|maximum\s+age\s+of)\s*(\d+)'
    r'|(?:younger\s+than|under|below|less\s+than)\s+(\d+)\s*' + _UNIT +
    r'|(\d+)\s*' + _UNIT + r'?\s*(?:of\s+age\s*)?(?:and|or)\s+(?:younger|under|below|less)',
    re.IGNORECASE
)
# "aged 18 years" with no direction: eligibility lists almost always mean a minimum age
AGE_BARE_RE = re.compile(r'\bag(?:e|ed)\s+(\d+)\s*' + _UNIT, re.IGNORECASE)


def _first_group(m):
    return float(next(g for g in m.groups() if g is not None))


SEX_RE = re.compile(
    r'\b(male|female|men|women|both\s+sexes?)\b',
    re.IGNORECASE
)
SEX_ONLY_RE = re.compile(
    r'\b('
    r'males?\s+only|females?\s+only|men\s+only|women\s+only|'
    r'only\s+(?:male|female|men|women)\s+(?:patients?|subjects?|participants?)?|'
    r'(?:female|male|women|men)\s+(?:patients?|subjects?|participants?)\s+only'
    r')\b',
    re.IGNORECASE
)

LAB_PATTERNS = [
    # HbA1c between X% and Y%
    (re.compile(
        r'(?:HbA1c|hemoglobin\s+a1c|glycated\s+hemoglobin)\s+'
        r'(?:between\s+)?(\d+(?:\.\d+)?)\s*%?\s*(?:and|-|to)\s*(\d+(?:\.\d+)?)\s*%?',
        re.IGNORECASE
    ), "hemoglobin_a1c", "between"),
    # HbA1c >= X
    (re.compile(
        r'(?:HbA1c|hemoglobin\s+a1c)\s*(?:of\s+)?(?:≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)',
        re.IGNORECASE
    ), "hemoglobin_a1c", "single_op"),
    # creatinine <=/</>/>= X
    (re.compile(
        r'(?:serum\s+)?creatinine\s*(?:of\s+)?(?:≤|<=|<|≥|>=|>)\s*(\d+(?:\.\d+)?)',
        re.IGNORECASE
    ), "creatinine", "single_op"),
    # eGFR >= X
    (re.compile(
        r'eGFR\s*(?:of\s+)?(?:≥|>=|>|≤|<=|<)\s*(\d+(?:\.\d+)?)',
        re.IGNORECASE
    ), "egfr", "single_op"),
    # ALT / AST <= X x ULN
    (re.compile(
        r'(?:ALT|AST|alanine\s+aminotransferase|aspartate\s+aminotransferase)\s*'
        r'(?:≤|<=|<)\s*(\d+(?:\.\d+)?)\s*[x×]?\s*(?:ULN|upper\s+limit\s+of\s+normal)',
        re.IGNORECASE
    ), "alt_or_ast", "uln_multiple"),
]

PREGNANCY_RE = re.compile(
    r'(?:not|non-?)\s*pregnant|pregnant\s+women\s+excluded|'
    r'negative\s+pregnancy\s+test|not\s+breastfeeding',
    re.IGNORECASE
)
PREGNANCY_REQUIRED_RE = re.compile(r'\bpregnant\b', re.IGNORECASE)


def _op_from_match(text_fragment: str) -> str:
    if "≤" in text_fragment or "<=" in text_fragment:
        return "<="
    if "≥" in text_fragment or ">=" in text_fragment:
        return ">="
    if "<" in text_fragment:
        return "<"
    if ">" in text_fragment:
        return ">"
    return "=="


def normalize_clause(text: str) -> list[dict]:
    """
    Attempt deterministic normalization of a clause.
    Returns list of constraint dicts, or empty list if none found.
    """
    constraints = []

    # Age: range, then one-sided bounds. Only attempted when the clause is about age.
    if _AGE_CTX_RE.search(text):
        m = AGE_RANGE_RE.search(text)
        if m:
            lo, hi = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(3), m.group(4))
            constraints.append({"type": "age", "min": float(lo), "max": float(hi)})
        else:
            m_min = AGE_MIN_RE.search(text)
            m_max = AGE_MAX_RE.search(text)
            if m_min:
                constraints.append({"type": "age", "min": _first_group(m_min)})
            if m_max:
                constraints.append({"type": "age", "max": _first_group(m_max)})
            if not m_min and not m_max:
                m_bare = AGE_BARE_RE.search(text)
                if m_bare:
                    constraints.append({"type": "age", "min": float(m_bare.group(1))})

    # Sex
    if SEX_ONLY_RE.search(text):
        m = SEX_ONLY_RE.search(text)
        val = m.group(1).lower()
        # Use word-boundary-safe checks to avoid "female" matching "male"
        is_female = bool(re.search(r'\bfemale\b|\bwomen\b|\bwoman\b', val))
        is_male = bool(re.search(r'\bmale\b|\bmen\b|\bman\b', val))
        if is_female and not is_male:
            sex = "female"
        elif is_male and not is_female:
            sex = "male"
        else:
            # Both or ambiguous → allow all
            constraints.append({"type": "sex", "allowed": ["male", "female", "other"]})
            sex = None
        if sex:
            constraints.append({"type": "sex", "allowed": [sex]})
    elif re.search(r'\b(all sexes?|both sexes?|male or female)\b', text, re.IGNORECASE):
        constraints.append({"type": "sex", "allowed": ["male", "female", "other"]})

    # Labs
    for pattern, test_name, mode in LAB_PATTERNS:
        m = pattern.search(text)
        if m:
            if mode == "between":
                constraints.append({
                    "type": "lab", "test": test_name, "op": ">=", "value": float(m.group(1))
                })
                constraints.append({
                    "type": "lab", "test": test_name, "op": "<=", "value": float(m.group(2))
                })
            elif mode == "single_op":
                op = _op_from_match(text[max(0, m.start()-5):m.end()])
                constraints.append({
                    "type": "lab", "test": test_name, "op": op, "value": float(m.group(1))
                })
            elif mode == "uln_multiple":
                constraints.append({
                    "type": "lab",
                    "test": test_name,
                    "op": "<=",
                    "value": float(m.group(1)),
                    "unit": "x_ULN",
                })

    # Pregnancy exclusion
    if PREGNANCY_RE.search(text):
        constraints.append({
            "type": "exclusion",
            "text": "Not pregnant / not breastfeeding",
        })

    return constraints


# ─── Rule Verifier ───────────────────────────────────────────────────────────

class RuleVerificationResult:
    def __init__(self):
        self.passed: list[dict] = []
        self.failed: list[dict] = []
        self.uncertain: list[dict] = []

    @property
    def rule_score(self) -> float:
        if self.failed:
            return 0.0
        if self.uncertain:
            return 0.5
        if self.passed:
            return 1.0
        return 0.5  # No deterministic constraints found

    @property
    def has_hard_exclusion(self) -> bool:
        return bool(self.failed)

    def summary(self) -> dict:
        return {
            "rule_score": self.rule_score,
            "has_hard_exclusion": self.has_hard_exclusion,
            "passed": self.passed,
            "failed": self.failed,
            "uncertain": self.uncertain,
        }


def verify_constraints(
    constraints: list[dict],
    patient: "PatientInput",  # type: ignore
    is_inclusion: bool = True,
) -> RuleVerificationResult:
    """
    Evaluate a list of normalized constraints against a patient.
    For inclusion: failing means patient is excluded.
    For exclusion: passing means patient is excluded.
    """
    result = RuleVerificationResult()

    for c in constraints:
        ctype = c.get("type")

        if ctype == "age":
            age = patient.age
            age_min = c.get("min")
            age_max = c.get("max")
            if age_min is not None and age < age_min:
                if is_inclusion:
                    result.failed.append({**c, "reason": f"Patient age {age} < min {age_min}"})
                else:
                    result.passed.append(c)
                continue
            if age_max is not None and age > age_max:
                if is_inclusion:
                    result.failed.append({**c, "reason": f"Patient age {age} > max {age_max}"})
                else:
                    result.passed.append(c)
                continue
            result.passed.append(c)

        elif ctype == "sex":
            allowed = c.get("allowed", [])
            if allowed and patient.sex.lower() not in [s.lower() for s in allowed]:
                if is_inclusion:
                    result.failed.append({**c, "reason": f"Patient sex '{patient.sex}' not in {allowed}"})
                else:
                    result.passed.append(c)
            else:
                result.passed.append(c)

        elif ctype == "lab":
            test = c.get("test", "")
            op = c.get("op", "==")
            threshold = c.get("value")
            if threshold is None:
                result.uncertain.append({**c, "reason": "No threshold value"})
                continue

            # Map test name to patient labs
            labs_dict = patient.labs.model_dump()
            patient_val = labs_dict.get(test)

            if patient_val is None:
                result.uncertain.append({**c, "reason": f"Lab '{test}' not available for patient"})
                continue

            ops = {">=": patient_val >= threshold, "<=": patient_val <= threshold,
                   ">": patient_val > threshold, "<": patient_val < threshold,
                   "==": patient_val == threshold}
            passes = ops.get(op, True)

            if is_inclusion:
                if passes:
                    result.passed.append(c)
                else:
                    result.failed.append({**c, "reason": f"Patient {test}={patient_val} fails {op}{threshold}"})
            else:
                # Exclusion: if patient meets exclusion criterion → exclude
                if passes:
                    result.failed.append({**c, "reason": f"Patient meets exclusion: {test}={patient_val} {op}{threshold}"})
                else:
                    result.passed.append(c)

        elif ctype == "exclusion":
            # Text-based exclusion: always uncertain unless we have structured data
            result.uncertain.append({**c, "reason": "Free-text exclusion — requires LLM or clinician review"})

        else:
            result.uncertain.append({**c, "reason": f"Constraint type '{ctype}' not deterministically evaluable"})

    return result
