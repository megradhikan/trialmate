"""
End-to-end regression test: the hybrid ranker must clearly beat a random
ranking and a rules-only ranking on registry-derived labels
(see evaluation/run_ablation.py and evaluation/build_registry_labels.py).

Thresholds sit well below the measured values (hybrid P@5 ~0.48, random ~0.22,
rule-only ~0.25 with all-MiniLM-L6-v2) so the test catches real regressions
without flaking.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _deps_available() -> bool:
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _deps_available(), reason="faiss / sentence_transformers not installed")
def test_hybrid_beats_baselines(tmp_path):
    out = tmp_path / "ablation.json"
    env = {**os.environ, "LLM_PROVIDER": "mock"}
    subprocess.run(
        [sys.executable, "-m", "evaluation.run_ablation", "--model", "all-MiniLM-L6-v2",
         "--labels", "registry", "--out", str(out)],
        cwd=ROOT, env=env, check=True, capture_output=True, timeout=900,
    )
    res = json.loads(out.read_text())["results"]
    hybrid, rule, rand = res["hybrid"], res["rule_only"], res["random"]
    assert hybrid["p@5"] >= 0.35, hybrid
    assert hybrid["p@5"] > rand["p@5"] + 0.10, (hybrid, rand)
    assert hybrid["p@5"] > rule["p@5"] + 0.10, (hybrid, rule)
