"""retrieve -> abstain gate -> generate."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from .generation import Generator
from .retriever import Hit, Retriever
from .text import WORD, tokenize

ABSTAIN_MESSAGE = (
    "I couldn't find anything about that in the indexed project sources, so I won't guess. "
    "Try asking about the projects' features, code or design."
)


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    citations: list[list[int]] = field(default_factory=list)
    abstained: bool = False
    generator: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)

    def cited_hits(self) -> list[Hit]:
        seen: list[int] = []
        for group in self.citations:
            seen.extend(i for i in group if i not in seen)
        return [self.hits[i] for i in seen]


class RagPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, *, top_k: int = 6,
                 min_coverage: float = 0.5, min_dense: float = 0.75):
        """The abstain gate passes when at least `min_coverage` of the question's IDF-weighted terms appear in
        the retrieved chunks, or when the best dense similarity reaches `min_dense`."""
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k
        self.min_coverage = min_coverage
        self.min_dense = min_dense

    def term_coverage(self, question: str, hits: list[Hit]) -> float:
        """IDF-weighted share of the question's words that appear in the retrieved chunks.

        Counted per *word*, not per token: `calcVat` expands to calcvat/calc/vat and must not outweigh one
        off-topic word. A word is covered if its full form or all of its parts appear."""
        idf = self.retriever.index.bm25.idf
        present = set().union(*(tokenize(h.chunk.indexed_text()) for h in hits)) if hits else set()
        # Unknown terms get the maximum IDF: a word absent from the whole corpus is strong evidence of off-topic.
        max_idf = max(idf.values(), default=1.0)
        total = covered = 0.0
        seen: set[str] = set()
        for word in WORD.findall(question):
            toks = tokenize(word)
            if not toks or toks[0] in seen:
                continue
            seen.add(toks[0])
            weight = max(idf.get(t, max_idf) for t in toks)
            total += weight
            if toks[0] in present or all(t in present for t in toks[1:] or toks):
                covered += weight
        return covered / total if total else 0.0

    def answerable(self, question: str, hits: list[Hit]) -> tuple[bool, dict]:
        if not hits:
            return False, {"coverage": 0.0, "max_dense": 0.0}
        coverage = self.term_coverage(question, hits)
        max_dense = max((h.dense or -math.inf) for h in hits)
        ok = coverage >= self.min_coverage or max_dense >= self.min_dense
        return ok, {"coverage": round(coverage, 3), "max_dense": round(float(max_dense), 4)}

    def retrieve(self, question: str, k: int | None = None, mode: str | None = None) -> tuple[list[Hit], dict]:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k or self.top_k, mode=mode)
        ok, gate = self.answerable(question, hits)
        gate = {**gate, "answerable": ok, "mode": mode or self.retriever.mode, "retrieval_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return hits, gate

    def generate(self, question: str, hits: list[Hit], gate: dict) -> Iterator[dict]:
        if not gate["answerable"]:
            yield {"type": "token", "text": ABSTAIN_MESSAGE}
            yield {"type": "done", "generator": "abstain", "abstained": True}
            return
        t0 = time.perf_counter()
        for event in self.generator.stream(question, hits):
            if event["type"] == "done":
                event = {**event, "generation_ms": round((time.perf_counter() - t0) * 1000, 1)}
            yield event

    def stream(self, question: str, k: int | None = None, mode: str | None = None) -> Iterator[dict]:
        hits, gate = self.retrieve(question, k, mode)
        yield {"type": "sources", "hits": [h.to_dict() for h in hits], "gate": gate}
        yield from self.generate(question, hits, gate)

    def answer(self, question: str, k: int | None = None) -> Answer:
        """Non-streaming convenience wrapper (CLI, tests, eval)."""
        hits, gate = self.retrieve(question, k)
        ans = Answer(question=question, text="", hits=hits, meta={"gate": gate})
        parts: list[str] = []
        for ev in self.generate(question, hits, gate):
            if ev["type"] == "token":
                parts.append(ev["text"])
            elif ev["type"] == "cite":
                ans.citations.append(ev["hits"])
            elif ev["type"] == "done":
                ans.generator = ev.get("generator", "")
                ans.abstained = bool(ev.get("abstained"))
                ans.meta.update({k2: v for k2, v in ev.items() if k2 not in ("type", "generator", "abstained")})
            elif ev["type"] == "error":
                ans.error = ev["message"]
        ans.text = "".join(parts)
        return ans
