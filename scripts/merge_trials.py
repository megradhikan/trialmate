"""
scripts/merge_trials.py
Merges multiple ClinicalTrials.gov JSON downloads into one file.

Usage:
    python scripts/merge_trials.py --inputs data/raw/diabetes.json data/raw/hypertension.json data/raw/lungcancer.json --output data/raw/all_trials.json
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", nargs="+", required=True)
parser.add_argument("--output", default="data/raw/all_trials.json")
args = parser.parse_args()

all_studies = []
seen = set()
for path in args.inputs:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    studies = data.get("studies", [])
    for s in studies:
        nct = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "")
        if nct and nct not in seen:
            seen.add(nct)
            all_studies.append(s)
    print(f"{path}: {len(studies)} studies")

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"studies": all_studies}, f)
print(f"\nMerged {len(all_studies)} unique trials → {args.output}")