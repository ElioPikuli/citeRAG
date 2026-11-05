"""Retrieval evaluation on a golden set.

golden.jsonl rows: {"question": str, "relevant": [source path, ...]}. Relevance is judged per *file*, so the
golden set survives chunking changes. A row with an empty `relevant` list is an unanswerable question: it is
excluded from retrieval metrics and used to score the abstain gate instead.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .retriever import Mode, Retriever


@dataclass
class GoldenItem:
    question: str
    relevant: list[str]
    slice: str = "all"


@dataclass
class RetrievalReport:
    mode: str
    n: int
    hit_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    per_query: list[dict] = field(default_factory=list)


def load_golden(path: str | Path) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                items.append(GoldenItem(row["question"], list(row.get("relevant", [])), row.get("slice", "all")))
    return items


def ranked_sources(sources: list[str]) -> list[str]:
    """Distinct sources in order of first appearance (several chunks of one file count once)."""
    seen: list[str] = []
    for s in sources:
        if s not in seen:
            seen.append(s)
    return seen


def score_query(retrieved_sources: list[str], relevant: set[str], k: int) -> dict:
    top = retrieved_sources[:k]
    first = next((r for r, s in enumerate(top, start=1) if s in relevant), None)
    distinct = ranked_sources(top)
    found = [s for s in distinct if s in relevant]
    dcg = sum(1.0 / math.log2(r + 1) for r, s in enumerate(distinct, start=1) if s in relevant)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return {
        "hit": first is not None,
        "recall": len(found) / len(relevant),
        "rr": 1.0 / first if first else 0.0,
        "ndcg": dcg / idcg if idcg else 0.0,
    }


def evaluate(retriever: Retriever, golden: list[GoldenItem], mode: Mode = "hybrid", k: int = 5) -> RetrievalReport:
    rows = []
    for item in golden:
        if not item.relevant:
            continue
        hits = retriever.search(item.question, k, mode=mode)
        m = score_query([h.chunk.source for h in hits], set(item.relevant), k)
        rows.append({"question": item.question, "slice": item.slice, **m,
                     "retrieved": [h.chunk.label for h in hits]})
    return _aggregate(mode, rows)


def _aggregate(mode: str, rows: list[dict]) -> RetrievalReport:
    n = len(rows) or 1
    return RetrievalReport(
        mode=mode,
        n=len(rows),
        hit_at_k=sum(r["hit"] for r in rows) / n,
        recall_at_k=sum(r["recall"] for r in rows) / n,
        mrr=sum(r["rr"] for r in rows) / n,
        ndcg_at_k=sum(r["ndcg"] for r in rows) / n,
        per_query=rows,
    )


def evaluate_gate(pipeline, golden: list[GoldenItem]) -> dict:
    """How well the abstain gate separates answerable from unanswerable questions."""
    tp = fp = tn = fn = 0  # positive = "answerable"
    for item in golden:
        _, gate = pipeline.retrieve(item.question)
        if item.relevant:
            tp += gate["answerable"]
            fn += not gate["answerable"]
        else:
            fp += gate["answerable"]
            tn += not gate["answerable"]
    return {
        "answered_when_answerable": tp / ((tp + fn) or 1),
        "abstained_when_unanswerable": tn / ((tn + fp) or 1),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def by_slice(report: RetrievalReport) -> dict[str, RetrievalReport]:
    slices: dict[str, list[dict]] = {}
    for row in report.per_query:
        slices.setdefault(row["slice"], []).append(row)
    return {name: _aggregate(report.mode, rows) for name, rows in slices.items()}


def format_table(results: dict[str, RetrievalReport], k: int) -> str:
    """Markdown table: one row per (slice, retriever), overall first."""
    lines = [
        f"| questions | retriever | Hit@{k} | Recall@{k} | MRR | nDCG@{k} |",
        "|---|---|---|---|---|---|",
    ]

    def row(label: str, mode: str, r: RetrievalReport) -> str:
        return (f"| {label} | {mode} | {r.hit_at_k:.3f} | {r.recall_at_k:.3f} | {r.mrr:.3f} | "
                f"{r.ndcg_at_k:.3f} |")

    for mode, r in results.items():
        lines.append(row(f"all ({r.n})", mode, r))
    slice_names = sorted({q["slice"] for r in results.values() for q in r.per_query})
    if len(slice_names) > 1:
        for name in slice_names:
            for mode, r in results.items():
                s = by_slice(r)[name]
                lines.append(row(f"{name} ({s.n})", mode, s))
    return "\n".join(lines)
