"""Okapi BM25 over pre-tokenized documents. Sparse postings, no dependencies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        # Lucene-style IDF: always positive, so very common terms still count a little.
        self.idf = {t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = [0.0] * self.n
        if not self.n or not self.avgdl:
            return out
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def top(self, query: list[str], k: int) -> list[tuple[int, float]]:
        s = self.scores(query)
        ranked = sorted((i for i in range(self.n) if s[i] > 0), key=lambda i: -s[i])
        return [(i, s[i]) for i in ranked[:k]]
