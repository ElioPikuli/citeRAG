import json
import os
from pathlib import Path

import pytest

from citerag.evaluation import by_slice, evaluate, evaluate_gate, load_golden, score_query

ROOT = Path(__file__).resolve().parent.parent


def test_score_query_metrics():
    m = score_query(["a", "b", "a", "c"], {"b", "c"}, k=4)
    assert m["hit"] and m["rr"] == 0.5 and m["recall"] == 1.0
    # distinct ranking a, b, c: gains at ranks 2 and 3; ideal at 1 and 2
    import math
    assert m["ndcg"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert score_query(["x"], {"b"}, k=1) == {"hit": False, "recall": 0.0, "rr": 0.0, "ndcg": 0.0}


def test_evaluate_on_fixture_corpus(pipeline, tmp_path):
    golden = tmp_path / "g.jsonl"
    rows = [
        {"question": "calcLaborDays", "relevant": ["proj/calc.js"], "slice": "identifier"},
        {"question": "model accuracy", "relevant": ["site/index.html"], "slice": "natural"},
        {"question": "capital of Australia", "relevant": [], "slice": "unanswerable"},
    ]
    golden.write_text("\n".join(json.dumps(r) for r in rows))
    items = load_golden(golden)
    report = evaluate(pipeline.retriever, items, mode="hybrid", k=3)
    assert report.n == 2 and report.hit_at_k == 1.0
    assert set(by_slice(report)) == {"identifier", "natural"}
    gate = evaluate_gate(pipeline, items)
    assert gate["tn"] == 1 and gate["tp"] == 2


@pytest.mark.skipif(
    not (ROOT / "index" / "manifest.json").exists() or os.environ.get("CITERAG_SKIP_REAL") == "1",
    reason="real index not built (run `citerag ingest`)",
)
def test_regression_floor_on_real_index():
    """Guards retrieval quality on the demo corpus. Uses the real embedding model."""
    from citerag.config import Settings
    from citerag.engine import load_pipeline

    settings = Settings.load(ROOT / "citerag.toml")
    p = load_pipeline(settings)
    golden = load_golden(ROOT / "eval" / "golden.jsonl")
    needed = {s for item in golden for s in item.relevant}
    if not needed <= set(p.retriever.index.files):
        pytest.skip("index does not cover the full demo corpus (run scripts/fetch_projects.py, then ingest)")
    report = evaluate(p.retriever, golden, k=5)
    assert report.hit_at_k >= 0.95
    assert report.mrr >= 0.85
    gate = evaluate_gate(p, golden)
    assert gate["abstained_when_unanswerable"] == 1.0
    assert gate["answered_when_answerable"] >= 0.95
