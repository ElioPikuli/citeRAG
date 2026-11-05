"""Embedders. All return L2-normalized float32 rows, so cosine similarity is a dot product."""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashingEmbedder:
    """Feature-hashed bag of words and character trigrams. No model download; deterministic.
    Used by the test suite and as a zero-dependency fallback. Not semantic: it only captures lexical overlap."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = tokenize(text)
        feats += [f"#{t[i:i + 3]}" for t in feats for i in range(max(len(t) - 2, 1))]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


class FastEmbedEmbedder:
    """Local ONNX sentence embeddings via fastembed (default BAAI/bge-small-en-v1.5, 384-d)."""

    # bge models are trained with this instruction prefix on the query side only.
    QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # imported lazily: heavy, and tests don't need it

        self.model_name = model
        # CITERAG_THREADS caps ONNX threads. On a CPU-quota host (Render free: 0.1 CPU) the default of one
        # thread per visible core causes heavy throttling; 1 thread is several times faster there.
        threads = int(os.environ["CITERAG_THREADS"]) if os.environ.get("CITERAG_THREADS") else None
        self._model = TextEmbedding(model, threads=threads)
        self.dim = int(next(iter(self._model.embed(["dimension probe"]))).shape[0])
        self.name = f"fastembed:{model}"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack(list(self._model.embed(texts, batch_size=64))))

    def embed_query(self, text: str) -> np.ndarray:
        q = self.QUERY_PREFIX.get(self.model_name, "") + text
        return _normalize(np.asarray(next(iter(self._model.embed([q]))))[None, :])[0]


def make_embedder(spec: str) -> Embedder:
    kind, _, arg = spec.partition(":")
    if kind == "hashing":
        return HashingEmbedder(int(arg) if arg else 512)
    if kind == "fastembed":
        return FastEmbedEmbedder(arg or "BAAI/bge-small-en-v1.5")
    raise ValueError(f"unknown embedder spec: {spec!r} (use 'fastembed:<model>' or 'hashing:<dim>')")
