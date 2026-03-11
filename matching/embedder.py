"""
matching/embedder.py — Build and load FAISS embedding index for trial clauses
matching/retriever.py — Retrieve top-K relevant clauses for a patient
"""
from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

# ─── Embedding backend selection ─────────────────────────────────────────────

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_embedder():
    """Return embedding function based on provider env var."""
    if EMBEDDING_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

        def embed_fn(texts: list[str]) -> np.ndarray:
            response = client.embeddings.create(input=texts, model=model)
            return np.array([e.embedding for e in response.data], dtype=np.float32)

        return embed_fn
    else:
        # Default: SentenceTransformers (local, no API key)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)

        def embed_fn(texts: list[str]) -> np.ndarray:
            return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        return embed_fn


# ─── Index Builder ────────────────────────────────────────────────────────────

class ClauseIndex:
    """FAISS index over trial clause embeddings + metadata lookup."""

    def __init__(self):
        self.embed_fn = None
        self.index = None  # faiss index
        self.clause_meta: list[dict] = []  # parallel to FAISS vectors

    def build(self, trials: list[dict], output_dir: str = "data/processed/faiss_index"):
        """Build FAISS index from list of parsed trial dicts."""
        import faiss

        self.embed_fn = get_embedder()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect all clauses
        all_clauses = []
        for trial in trials:
            for clause in trial.get("clauses", []):
                all_clauses.append({
                    "nct_id": trial["nct_id"],
                    "clause_id": clause["clause_id"],
                    "text": clause["text"],
                    "type": clause.get("type", "other"),
                    "is_inclusion": clause.get("is_inclusion", True),
                })

        if not all_clauses:
            logger.warning("No clauses found in trials list.")
            return

        logger.info(f"Embedding {len(all_clauses)} clauses...")
        texts = [c["text"] for c in all_clauses]
        embeddings = self.embed_fn(texts)  # (N, D)
        embeddings = embeddings.astype(np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product = cosine on normalized vecs
        self.index.add(embeddings)
        self.clause_meta = all_clauses

        # Save
        faiss.write_index(self.index, str(Path(output_dir) / "clauses.faiss"))
        with open(Path(output_dir) / "clause_meta.pkl", "wb") as f:
            pickle.dump(self.clause_meta, f)
        logger.info(f"Index saved to {output_dir}")

    def load(self, index_dir: str = "data/processed/faiss_index"):
        """Load previously built FAISS index."""
        import faiss

        self.embed_fn = get_embedder()
        self.index = faiss.read_index(str(Path(index_dir) / "clauses.faiss"))
        with open(Path(index_dir) / "clause_meta.pkl", "rb") as f:
            self.clause_meta = pickle.load(f)
        logger.info(f"Loaded index with {self.index.ntotal} vectors.")

    def search(self, query_text: str, top_k: int = 20) -> list[dict]:
        """Retrieve top-K clause metadata dicts by semantic similarity."""
        qvec = self.embed_fn([query_text]).astype(np.float32)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
        scores, indices = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = dict(self.clause_meta[idx])
            meta["similarity"] = float(score)
            results.append(meta)
        return results


# ─── CLI for building index ───────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS clause index")
    parser.add_argument("--trials", default="data/processed/trials.json")
    parser.add_argument("--output", default="data/processed/faiss_index")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.trials) as f:
        trials = json.load(f)

    idx = ClauseIndex()
    idx.build(trials, output_dir=args.output)
