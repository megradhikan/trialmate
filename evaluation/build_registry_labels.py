"""
evaluation/build_registry_labels.py
Derive relevance labels from ClinicalTrials.gov *structured* registry fields
(not from the free-text eligibility criteria the system parses):

    trial is relevant to patient  <=>  one of the trial's listed conditions
    matches one of the patient's condition families (type 2 diabetes,
    hypertension, lung cancer)  AND  the patient's age is within the registry
    minimumAge/maximumAge  AND  the registry sex field admits the patient's sex.

These are *silver* labels: they check topical + demographic fit only (no labs,
no medications, no exclusion criteria), so they measure how well the system
recovers structured eligibility from text, not full clinical eligibility.

Usage:
    python -m evaluation.build_registry_labels
"""
from __future__ import annotations
import json
import re
from pathlib import Path

FAMILIES = {
    "type 2 diabetes": (
        re.compile(r"diabet", re.I),
        re.compile(r"type 1|type i\b|gestational|insipidus", re.I),
        re.compile(r"diabet", re.I),
    ),
    "hypertension": (
        re.compile(r"hypertens", re.I),
        re.compile(r"pulmonary|portal|ocular|intracranial|intraocular|pregnan|preeclampsia", re.I),
        re.compile(r"hypertens", re.I),
    ),
    "lung cancer": (
        re.compile(r"lung (cancer|carcinoma|adenocarcinoma|neoplasm)|nsclc|sclc|non-?small.?cell|small.?cell lung", re.I),
        None,
        re.compile(r"lung (cancer|carcinoma|adenocarcinoma|neoplasm)|nsclc|sclc|non-?small.?cell|small.?cell lung", re.I),
    ),
}
_UNIT_YEARS = {"year": 1, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365, "hour": 0, "minute": 0}


def _years(text):
    if not text:
        return None
    m = re.match(r"\s*(\d+)\s*(\w+)", text)
    if not m:
        return None
    return int(m.group(1)) * _UNIT_YEARS.get(m.group(2).lower().rstrip("s"), 1)


def trial_families(conditions: list[str]) -> set[str]:
    out = set()
    for fam, (inc, exc, _) in FAMILIES.items():
        for c in conditions:
            if inc.search(c) and not (exc and exc.search(c)):
                out.add(fam)
    return out


def patient_families(conditions: list[str]) -> set[str]:
    return {fam for fam, (_, _, pat) in FAMILIES.items() if any(pat.search(c) for c in conditions)}


def main():
    raw = json.load(open("data/raw/all_trials.json"))["studies"]
    trials = []
    for s in raw:
        p = s["protocolSection"]
        em = p.get("eligibilityModule", {})
        trials.append({
            "nct_id": p["identificationModule"]["nctId"],
            "families": trial_families(p.get("conditionsModule", {}).get("conditions", [])),
            "min_age": _years(em.get("minimumAge")),
            "max_age": _years(em.get("maximumAge")),
            "sex": em.get("sex", "ALL").upper(),
        })

    patients = json.load(open("data/samples/patients.json"))
    out = []
    for pt in patients:
        fams = patient_families(pt["conditions"])
        if not fams:
            continue
        rel = [
            t["nct_id"] for t in trials
            if t["families"] & fams
            and (t["min_age"] is None or pt["age"] >= t["min_age"])
            and (t["max_age"] is None or pt["age"] <= t["max_age"])
            and t["sex"] in ("ALL", pt["sex"].upper())
        ]
        if rel:
            out.append({"patient_id": pt["patient_id"], "relevant_nct_ids": rel})

    Path("evaluation/registry_labels.json").write_text(json.dumps(out, indent=1) + "\n")
    sizes = [len(o["relevant_nct_ids"]) for o in out]
    print(f"{len(out)}/{len(patients)} patients with >=1 covered condition; "
          f"relevant trials per patient: min {min(sizes)}, mean {sum(sizes)/len(sizes):.1f}, max {max(sizes)}")


if __name__ == "__main__":
    main()
