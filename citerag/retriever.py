"""Hybrid retrieval: BM25 + dense cosine, fused with Reciprocal Rank Fusion, diversified with MMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .chunking import Chunk
from .embeddings import Embedder
from .store import Index
from .text import tokenize

Mode = Literal["hybrid", "bm25", "dense"]
RRF_K = 60


@dataclass
class Hit:
    chunk: Chunk
    score: float  # final ranking score (RRF for hybrid, raw score otherwise)
    bm25: float | None = None
    bm25_rank: int | None = None
    dense: float | None = None
    dense_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            **self.chunk.to_dict(),
            "label": self.chunk.label,
            "score": round(self.score, 5),
            "bm25": None if self.bm25 is None else round(self.bm25, 3),
            "bm25_rank": self.bm25_rank,
            "dense": None if self.dense is None else round(self.dense, 4),
            "dense_rank": self.dense_rank,
        }


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


def mmr(candidates: list[int], relevance: dict[int, float], vectors: np.ndarray, k: int, lam: float) -> list[int]:
    """Maximal Marginal Relevance: trade relevance against similarity to what is already selected, so the
    context window isn't spent on near-duplicate chunks (e.g. overlapping windows of one section)."""
    if not candidates:
        return []
    top = max(relevance[c] for c in candidates) or 1.0
    rel = {c: relevance[c] / top for c in candidates}
    selected: list[int] = []
    remaining = list(candidates)
    while remaining and len(selected) < k:
        if selected:
            sims = vectors[remaining] @ vectors[selected].T
            redundancy = sims.max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        scores = [lam * rel[c] - (1 - lam) * float(r) for c, r in zip(remaining, redundancy)]
        best = int(np.argmax(scores))
        selected.append(remaining.pop(best))
    return selected


class Retriever:
    def __init__(self, index: Index, embedder: Embedder, *, candidates: int = 30, mmr_lambda: float = 0.7,
                 mode: Mode = "hybrid"):
        if index.embedder_name != embedder.name:
            raise ValueError(f"index embedder {index.embedder_name!r} != query embedder {embedder.name!r}")
        self.index = index
        self.embedder = embedder
        self.candidates = candidates
        self.mmr_lambda = mmr_lambda
        self.mode = mode

    def search(self, query: str, k: int = 6, mode: Mode | None = None, diversify: bool = True) -> list[Hit]:
        mode = mode or self.mode
        if not query.strip() or not len(self.index):
            return []
        n_cand = max(self.candidates, k)

        bm25_top = self.index.bm25.top(tokenize(query), n_cand)
        bm25_score = {i: s for i, s in bm25_top}
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_top, start=1)}

        qv = self.embedder.embed_query(query)
        sims = self.index.vectors @ qv
        dense_order = np.argsort(-sims)[:n_cand].tolist()
        dense_rank = {i: r for r, i in enumerate(dense_order, start=1)}

        if mode == "bm25":
            ranking = {i: s for i, s in bm25_top}
        elif mode == "dense":
            ranking = {i: float(sims[i]) for i in dense_order}
        else:
            ranking = rrf([[i for i, _ in bm25_top], dense_order])

        ordered = sorted(ranking, key=lambda i: -ranking[i])[:n_cand]
        if diversify and self.mmr_lambda < 1.0:
            chosen = mmr(ordered, ranking, self.index.vectors, k, self.mmr_lambda)
        else:
            chosen = ordered[:k]
        return [
            Hit(
                chunk=self.index.chunks[i],
                score=ranking[i],
                bm25=bm25_score.get(i),
                bm25_rank=bm25_rank.get(i),
                dense=float(sims[i]),
                dense_rank=dense_rank.get(i),
            )
            for i in chosen
        ]
