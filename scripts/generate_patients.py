"""
scripts/generate_patients.py
Generates synthetic FHIR-like patient records for testing TrialMate.
No real patient data. Uses random seeds for reproducibility.

Usage:
    python scripts/generate_patients.py --count 200 --seed 42 --output data/samples/patients.json
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


CONDITION_POOL = [
    "type 2 diabetes", "hypertension", "coronary artery disease",
    "heart failure", "chronic kidney disease", "atrial fibrillation",
    "COPD", "asthma", "obesity", "hyperlipidemia", "hypothyroidism",
    "osteoporosis", "depression", "anxiety", "colorectal cancer",
    "breast cancer", "prostate cancer", "lung cancer", "multiple myeloma",
    "rheumatoid arthritis", "psoriasis", "Crohn's disease", "ulcerative colitis",
]

MEDICATION_POOL = [
    "metformin", "lisinopril", "atorvastatin", "amlodipine", "omeprazole",
    "levothyroxine", "insulin glargine", "semaglutide", "empagliflozin",
    "aspirin", "warfarin", "apixaban", "furosemide", "carvedilol",
    "ramipril", "dapagliflozin", "sitagliptin", "glimepiride",
    "prednisone", "methotrexate", "adalimumab", "pembrolizumab",
]


def generate_patient(patient_id: str, rng: random.Random) -> dict:
    age = rng.randint(18, 82)
    sex = rng.choice(["male", "female"])
    n_conditions = rng.randint(1, 4)
    conditions = rng.sample(CONDITION_POOL, n_conditions)
    n_meds = rng.randint(0, 5)
    medications = rng.sample(MEDICATION_POOL, n_meds)

    # Generate plausible labs
    is_diabetic = "type 2 diabetes" in conditions
    labs = {
        "hemoglobin_a1c": round(rng.uniform(5.5, 11.0) if is_diabetic else rng.uniform(4.8, 6.4), 1),
        "creatinine": round(rng.uniform(0.5, 2.8), 2),
        "alt": round(rng.uniform(10, 120), 1),
        "ast": round(rng.uniform(10, 100), 1),
        "egfr": round(rng.uniform(20, 120), 1),
        "wbc": round(rng.uniform(3.5, 15.0), 1),
        "platelets": round(rng.uniform(100, 450), 0),
        "bilirubin": round(rng.uniform(0.2, 2.5), 2),
    }

    # Pregnancy only applicable to females of reproductive age
    is_pregnant = None
    if sex == "female" and 18 <= age <= 45:
        is_pregnant = rng.random() < 0.05  # 5% chance

    ecog = rng.choices([0, 1, 2, 3], weights=[50, 30, 15, 5])[0]

    notes_templates = [
        f"Patient presents with {conditions[0]} requiring management.",
        f"Stable {conditions[0]}. Labs reviewed.",
        f"Follow-up for {', '.join(conditions[:2])}.",
        f"New referral. {conditions[0]} poorly controlled.",
    ]

    return {
        "patient_id": patient_id,
        "age": age,
        "sex": sex,
        "conditions": conditions,
        "medications": medications,
        "labs": labs,
        "notes": rng.choice(notes_templates),
        "icd_codes": [],
        "ecog_status": ecog,
        "is_pregnant": is_pregnant,
        "prior_therapies": [],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic patient records")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/samples/patients.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    patients = []
    for i in range(args.count):
        pid = f"P{i+1:04d}"
        patients.append(generate_patient(pid, rng))

    with open(args.output, "w") as f:
        json.dump(patients, f, indent=2)

    print(f"Generated {len(patients)} synthetic patients → {args.output}")
    # Show first patient as sample
    print("\nSample patient:")
    print(json.dumps(patients[0], indent=2))


if __name__ == "__main__":
    main()
