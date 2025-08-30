from __future__ import annotations
import os
from typing import List
import numpy as np

_EMBED_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

# Lazy imports to speed startup
_openai_client = None
_local_model = None

def _ensure_openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client

def _ensure_local():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model

def get_embedding(texts: List[str]) -> np.ndarray:
    """Return shape (N, D) embeddings for given texts."""
    if _EMBED_PROVIDER == "openai":
        client = _ensure_openai()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        mat = np.array([d.embedding for d in resp.data], dtype="float32")
        return mat
    else:
        model = _ensure_local()
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype="float32")
