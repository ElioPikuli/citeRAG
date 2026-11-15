from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import citerag.api as api_module
from citerag.api import create_app
from citerag.cli import main
from citerag.config import Settings
from citerag.tracing import (
    Tracer,
    estimate_cost,
    format_summary,
    load_traces,
    summarise,
    traced_search,
    traced_stream,
)

KEYS = {
    "trace_id", "ts", "source", "kind", "question", "question_chars", "mode", "k", "hits", "gate", "status",
    "generator", "model", "usage", "latency_ms", "cost_usd", "citations", "cite_events", "markers",
    "invalid_markers", "stop_reason", "error",
}
PRICING = {"claude-opus-5-5": {"input": 4.0, "output": 20.0}}


TIMING = ("retrieval_ms", "gate_ms", "generation_ms")


def _untimed(ev: dict) -> dict:
    """An event without its wall-clock fields, which differ between two runs of the same stream."""
    out = {k: v for k, v in ev.items() if k not in TIMING}
    if isinstance(out.get("gate"), dict):
        out["gate"] = {k: v for k, v in out["gate"].items() if k not in TIMING}
    return out


def read_traces(directory: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(directory.glob("*.jsonl")) for line in f.read_text().splitlines()]


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces", pricing=PRICING)


def test_successful_ask_trace(pipeline, tracer):
    q = "How are partial labor days billed? " + "labor " * 70
    expected = list(pipeline.stream(q))
    events = list(traced_stream(pipeline, tracer, q))
    assert [_untimed(e) for e in events] == [_untimed(e) for e in expected]
    (t,) = read_traces(tracer.directory)
    assert set(t) == KEYS
    assert t["status"] == "ok" and t["kind"] == "ask" and t["source"] == "api"
    assert t["citations"] == len({i for e in events if e["type"] == "cite" for i in e["hits"]})
    assert t["cite_events"] == sum(e["type"] == "cite" for e in events)
    lat = t["latency_ms"]
    assert all(lat[s] is not None for s in ("retrieve", "gate", "generate", "total"))
    assert lat["total"] >= lat["retrieve"]
    assert t["cost_usd"] == 0.0 and t["generator"] == "extractive"
    assert len(t["question"]) == 300 and t["question_chars"] == len(q)
    assert t["gate"]["answerable"] is True and t["hits"] and set(t["hits"][0]) == {
        "file", "start_line", "end_line", "score", "bm25_rank", "dense_rank", "dense"}
    assert t["error"] is None
    assert t["hits"][0]["file"] is not None and t["hits"][0]["start_line"] is not None
    assert t["k"] == pipeline.top_k


def test_effective_k_is_recorded(pipeline, tracer):
    list(traced_stream(pipeline, tracer, "labor days", k=3))
    assert read_traces(tracer.directory)[0]["k"] == 3


def test_abstained_ask_trace(pipeline, tracer):
    list(traced_stream(pipeline, tracer, "capital of Australia"))
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "abstained" and t["generator"] == "abstain"
    assert t["latency_ms"]["generate"] is None and t["cost_usd"] == 0.0
    assert t["gate"]["answerable"] is False


class ErrorGen:
    name = "ollama"

    def stream(self, question, hits):
        yield {"type": "token", "text": "partial"}
        yield {"type": "error", "message": "model fell over"}


class RaiseGen:
    name = "boom"

    def stream(self, question, hits):
        yield {"type": "token", "text": "partial"}
        raise RuntimeError("kaput")


def test_generator_error_event(pipeline, tracer):
    pipeline.generator = ErrorGen()
    events = list(traced_stream(pipeline, tracer, "labor days"))
    assert events[-1]["type"] == "error"
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "error" and t["error"] == {"type": "generator_error", "message": "model fell over"}
    assert t["cost_usd"] == 0.0  # ollama is local


def test_exception_is_recorded_and_reraised(pipeline, tracer):
    pipeline.generator = RaiseGen()
    with pytest.raises(RuntimeError):
        list(traced_stream(pipeline, tracer, "labor days"))
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "error" and t["error"]["type"] == "RuntimeError" and t["error"]["message"] == "kaput"
    assert t["cost_usd"] is None


def test_client_disconnect(pipeline, tracer):
    gen = traced_stream(pipeline, tracer, "labor days")
    assert next(gen)["type"] == "sources"
    gen.close()
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "cancelled" and t["error"] is None
    assert t["hits"] and t["latency_ms"]["total"] is not None


def test_close_after_done_keeps_ok(pipeline, tracer):
    gen = traced_stream(pipeline, tracer, "labor days")
    for ev in gen:
        if ev["type"] == "done":
            break
    gen.close()
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "ok" and t["error"] is None


def test_trace_is_written_before_done_is_yielded(pipeline, tracer):
    seen = []
    for ev in traced_stream(pipeline, tracer, "labor days"):
        if ev["type"] in ("done", "error"):
            seen = read_traces(tracer.directory)
    assert len(seen) == 1 and seen[0]["status"] == "ok"
    assert len(read_traces(tracer.directory)) == 1  # and not written twice


def test_stream_ending_without_done_is_incomplete(pipeline, tracer, monkeypatch):
    monkeypatch.setattr(pipeline, "stream", lambda q, k, mode: iter([{"type": "token", "text": "partial"}]))
    assert [e["type"] for e in traced_stream(pipeline, tracer, "labor days")] == ["token"]
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "error" and t["error"]["type"] == "incomplete"


def test_keyboard_interrupt_is_recorded_and_reraised(pipeline, tracer, monkeypatch):
    def stream(q, k, mode):
        yield {"type": "token", "text": "partial"}
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "stream", stream)
    with pytest.raises(KeyboardInterrupt):
        list(traced_stream(pipeline, tracer, "labor days"))
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "error" and t["error"]["type"] == "KeyboardInterrupt"


def test_traced_search_raising(pipeline, tracer, monkeypatch):
    def boom(*a, **kw):
        raise ValueError("bad index")

    monkeypatch.setattr(pipeline.retriever, "search", boom)
    with pytest.raises(ValueError):
        traced_search(pipeline, tracer, "labor", 2)
    (t,) = read_traces(tracer.directory)
    assert t["kind"] == "search" and t["status"] == "error" and t["error"]["type"] == "ValueError"
    assert t["hits"] == [] and t["latency_ms"]["total"] is not None


def test_bookkeeping_failure_does_not_break_request(pipeline, tracer, monkeypatch, caplog):
    events = [
        {"type": "sources", "hits": [{"source": "a.md", "score": "not-a-number"}], "gate": {"answerable": True}},
        {"type": "token", "text": "hi"},
        {"type": "done", "generator": "extractive"},
    ]
    monkeypatch.setattr(pipeline, "stream", lambda q, k, mode: iter(events))
    with caplog.at_level(logging.WARNING, logger="citerag.tracing"):
        out = list(traced_stream(pipeline, tracer, "labor days"))
    assert out == events
    assert "trace bookkeeping failed" in caplog.text
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "ok" and t["generator"] == "extractive" and t["hits"] == []


def test_concurrent_writes_stay_valid_json(tracer):
    def work(n):
        for i in range(25):
            tracer.write({"worker": n, "i": i, "pad": "x" * 500})

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = read_traces(tracer.directory)  # json.loads raises on any torn line
    assert len(rows) == 200


def test_write_failure_does_not_break_request(pipeline, tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    bad = Tracer(blocker / "sub")
    with caplog.at_level(logging.WARNING, logger="citerag.tracing"):
        events = list(traced_stream(pipeline, bad, "labor days"))
        hits = traced_search(pipeline, bad, "labor", 2)
    assert events[-1]["type"] == "done" and hits
    assert "could not write trace" in caplog.text


def test_disabled_and_none_write_nothing(pipeline, tmp_path):
    off = Tracer(tmp_path / "t", enabled=False)
    a = [e["type"] for e in traced_stream(pipeline, off, "labor days")]
    b = [e["type"] for e in traced_stream(pipeline, None, "labor days")]
    assert a == b and a[-1] == "done"
    assert traced_search(pipeline, None, "labor", 2)
    assert not (tmp_path / "t").exists()


def test_search_trace(pipeline, tracer):
    hits = traced_search(pipeline, tracer, "calcVat", 3, "bm25", source="cli")
    (t,) = read_traces(tracer.directory)
    assert set(t) == KEYS and t["kind"] == "search" and t["source"] == "cli" and t["gate"] is None
    assert t["mode"] == "bm25" and t["k"] == 3 and len(t["hits"]) == len(hits)
    assert t["hits"][0]["file"] == hits[0].chunk.source
    assert t["cost_usd"] == 0.0  # no model call; must not show up as an unknown cost


class ClaudeErrorGen:
    name = "claude"

    def stream(self, question, hits):
        yield {"type": "token", "text": "partial"}
        yield {"type": "error", "message": "overloaded"}


def test_claude_error_event_has_unknown_cost(pipeline, tracer):
    pipeline.generator = ClaudeErrorGen()
    list(traced_stream(pipeline, tracer, "labor days"))
    (t,) = read_traces(tracer.directory)
    assert t["status"] == "error" and t["generator"] == "claude" and t["cost_usd"] is None


def test_price_lookup_exact_then_longest_prefix():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    pricing = {"claude-opus-5": {"input": 5.0, "output": 25.0}, "claude-opus-5-5": {"input": 4.0, "output": 20.0}}
    assert estimate_cost("claude", "claude-opus-5-5", usage, pricing) == 4.0  # exact beats the shorter prefix
    assert estimate_cost("claude", "claude-opus-5-5-20261101", usage, pricing) == 4.0  # longest prefix
    assert estimate_cost("claude", "claude-opus-5-20261101", usage, pricing) == 5.0
    assert estimate_cost("claude", "claude-opus-50", usage, pricing) is None  # not a family member
    assert estimate_cost("claude", "claude-opus-5-5", usage, {"claude-opus-5": pricing["claude-opus-5"]}) == 5.0


@pytest.mark.parametrize("entry", ["cheap", None, {"input": 1.0}, {"input": "1", "output": 2}, {"input": True, "output": 1}])
def test_malformed_pricing_gives_unknown_cost(pipeline, tmp_path, entry):
    pipeline.generator = ClaudeLike()
    tr = Tracer(tmp_path / "t", pricing={"claude-opus-5-5": entry})
    list(traced_stream(pipeline, tr, "labor days"))
    (t,) = read_traces(tr.directory)
    assert t["cost_usd"] is None and t["status"] == "ok"


def test_estimate_cost():
    usage = {"input_tokens": 1000, "output_tokens": 500}
    assert estimate_cost("claude", "claude-opus-5-5", usage, PRICING) == pytest.approx(0.014)
    assert estimate_cost("claude", "claude-opus-5-5", usage, PRICING) == round(1000 * 4 / 1e6 + 500 * 20 / 1e6, 8)
    assert estimate_cost("claude", "unknown-model", usage, PRICING) is None
    assert estimate_cost("claude", "claude-opus-5-5", None, PRICING) is None
    assert estimate_cost("claude", "claude-opus-5-5", {"input_tokens": None, "output_tokens": 5}, PRICING) is None
    assert estimate_cost("ollama", "qwen", usage, PRICING) == 0.0
    assert estimate_cost("extractive", None, None, PRICING) == 0.0
    assert estimate_cost("abstain", None, None, PRICING) == 0.0


def _t(kind="ask", status="ok", gen="ollama", source="api", cites=0, cost=0.0, usage=None, error=None, ts="2026-01-01T00:00:00Z",
       **lat):
    latency = {"retrieve": None, "gate": None, "generate": None, "total": None, **lat}
    return {"kind": kind, "status": status, "generator": gen, "source": source, "citations": cites, "cost_usd": cost,
            "usage": usage, "error": error, "ts": ts, "latency_ms": latency, "question": "SECRET QUESTION"}


def test_summarise():
    traces = [
        _t(cites=2, usage={"input_tokens": 100, "output_tokens": 10}, cost=0.5, retrieve=10, gate=1, generate=100, total=111, ts="2026-01-01T00:00:01Z"),
        _t(cites=4, usage={"input_tokens": 50, "output_tokens": 5}, cost=None, retrieve=20, gate=3, generate=200, total=224, ts="2026-01-01T00:00:03Z"),
        _t(status="abstained", gen="abstain", retrieve=30, gate=2, total=32, ts="2026-01-01T00:00:02Z"),
        _t(kind="search", gen=None, source="cli", retrieve=40, total=40),
        _t(status="error", error={"type": "generator_error", "message": "m"}, cost=None, retrieve=50, gate=4, total=9000),
        _t(status="cancelled", retrieve=60, gate=5, total=7000),
        _t(kind="search", status="error", gen=None, error={"type": "ValueError", "message": "m"}, cost=None, retrieve=70, total=8000),
    ]
    s = summarise(traces)
    assert (s["requests"], s["asks"], s["searches"]) == (7, 5, 2)
    assert s["abstain_rate"] == 0.2
    assert s["error_rate"] == 2 / 7 and s["cancel_rate"] == 1 / 7
    # answer stages: only the asks with status ok/abstained; errors and cancellations don't skew them
    assert s["latency_ms"]["retrieve"] == {"p50": 20, "p95": 30, "n": 3}
    assert s["latency_ms"]["gate"] == {"p50": 2, "p95": 3, "n": 3}
    assert s["latency_ms"]["generate"] == {"p50": 100, "p95": 200, "n": 2}
    assert s["latency_ms"]["total"] == {"p50": 111, "p95": 224, "n": 3}
    assert s["search_latency_ms"] == {"p50": 40, "p95": 40, "n": 1}
    assert s["citations_per_answer"] == 3.0
    assert s["tokens"] == {"input": 150, "output": 15}
    assert s["cost_usd"] == {"total": 0.5, "unknown": 3}
    assert s["errors_by_type"] == {"generator_error": 1, "ValueError": 1}
    assert s["by_source"] == {"api": 6, "cli": 1} and s["by_generator"] == {"ollama": 4, "abstain": 1}
    assert s["first_ts"] == "2026-01-01T00:00:00Z" and s["last_ts"] == "2026-01-01T00:00:03Z"
    assert "SECRET" not in json.dumps(s) and "SECRET" not in format_summary(s)
    text = format_summary(s)
    assert "cancel rate 14.3%" in text and "search latency" in text


def test_summarise_empty():
    s = summarise([])
    assert s["requests"] == 0 and s["abstain_rate"] is None and s["error_rate"] is None
    assert s["cancel_rate"] is None and s["search_latency_ms"] == {"p50": None, "p95": None, "n": 0}
    assert s["citations_per_answer"] is None and s["latency_ms"]["total"] == {"p50": None, "p95": None, "n": 0}
    assert s["cost_usd"] == {"total": 0, "unknown": 0}
    assert "n/a" in format_summary(s)


def test_load_traces(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "2026-01-01.jsonl").write_text('{"n": 1}\nnot json\n{"n": 2}\n')
    (d / "2026-01-02.jsonl").write_text('{"n": 3}\n[1, 2]\n\n{"n": 4}\n')
    (d / "notes.jsonl").write_text('{"n": 99}\n')
    traces, bad = load_traces(d, 10)
    assert [t["n"] for t in traces] == [1, 2, 3, 4] and bad == 2
    traces, bad = load_traces(d, 3)
    assert [t["n"] for t in traces] == [2, 3, 4] and bad == 1
    assert load_traces(tmp_path / "missing") == ([], 0)


def test_load_traces_reads_only_a_bounded_tail(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    n = 400
    (d / "2026-01-01.jsonl").write_text("".join(json.dumps({"n": i, "pad": "é" * (i % 7)}, ensure_ascii=False) + "\n"
                                                  for i in range(n)), encoding="utf-8")
    size = (d / "2026-01-01.jsonl").stat().st_size
    for tail in (50, 333, 1000, 2047, 2048, size - 1, size, size + 10):
        traces, bad = load_traces(d, 10_000, tail_bytes=tail)
        ns = [t["n"] for t in traces]
        assert bad == 0, tail  # the cut first line is dropped, never counted
        assert ns == list(range(n - len(ns), n)), tail  # a contiguous suffix ending at the newest line
        assert (len(ns) == n) if tail >= size - 1 else (0 < len(ns) < n)
    assert [t["n"] for t in load_traces(d, 3, tail_bytes=1000)[0]] == [n - 3, n - 2, n - 1]


def test_load_traces_stops_once_last_is_collected(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    (d / "2026-01-02.jsonl").write_text('{"n": 2}\n{"n": 3}\n')
    (d / "2026-01-01.jsonl").write_text("garbage\n")
    traces, bad = load_traces(d, 2)
    assert [t["n"] for t in traces] == [2, 3] and bad == 0  # the older file was never opened


@pytest.mark.parametrize("bad_line", [
    {"status": 5}, {"kind": ["ask"]}, {"source": {}}, {"generator": 1.5}, {"citations": "3"}, {"citations": True},
    {"latency_ms": [1]}, {"latency_ms": {"total": "fast"}}, {"cost_usd": "free"}, {"error": "boom"},
    {"usage": [1, 2]}, {"ts": 12},
])
def test_load_traces_validates_field_types(tmp_path, bad_line):
    d = tmp_path / "t"
    d.mkdir()
    (d / "2026-01-01.jsonl").write_text(json.dumps({"status": "ok"}) + "\n" + json.dumps(bad_line) + "\n")
    traces, bad = load_traces(d)
    assert traces == [{"status": "ok"}] and bad == 1


def test_summarise_survives_odd_but_valid_traces():
    odd = [
        {}, {"error": {"type": ["x"]}, "status": "error"}, {"usage": {"input_tokens": "a", "output_tokens": None}},
        {"kind": "ask", "status": "ok", "citations": None, "latency_ms": {"total": None, "retrieve": 1}},
        {"kind": "search", "status": "ok", "latency_ms": {}, "cost_usd": None},
        {"kind": "ask", "status": "ok", "latency_ms": None, "usage": None, "error": None, "ts": None},
    ]
    s = summarise(odd)
    assert s["requests"] == 6 and s["errors_by_type"] == {"['x']": 1}
    json.dumps(s)
    format_summary(s)


class ClaudeLike:
    name = "claude"

    def stream(self, question, hits):
        yield {"type": "token", "text": "answer"}
        yield {"type": "cite", "hits": [0]}
        yield {"type": "done", "generator": "claude", "model": "claude-opus-5-5", "stop_reason": "end_turn",
               "usage": {"input_tokens": 2000, "output_tokens": 100}}


def test_api_traces_and_metrics(pipeline, tracer):
    pipeline.generator = ClaudeLike()
    client = TestClient(create_app(pipeline, tracer))
    assert client.get("/api/health").json()["tracing"] is True
    question = "How are partial labor days billed?"
    r = client.post("/api/ask", json={"question": question})
    assert r.status_code == 200
    assert client.get("/api/search", params={"q": "calcVat", "k": 2}).status_code == 200
    asks, searches = read_traces(tracer.directory)
    assert asks["kind"] == "ask" and asks["source"] == "api" and asks["cost_usd"] == round(2000 * 4e-6 + 100 * 20e-6, 8)
    assert searches["kind"] == "search" and searches["source"] == "api"
    m = client.get("/metrics", params={"last": 10}).json()
    assert m["enabled"] is True and m["requests"] == 2 and m["asks"] == 1 and m["malformed_lines"] == 0
    assert "question" not in json.dumps(m)
    raw = client.get("/metrics", params={"last": 10})
    assert question not in raw.text and "labor" not in raw.text
    assert asks["question"] == question  # it is in the trace file, just never in /metrics
    assert client.get("/metrics", params={"last": 0}).status_code == 422


def test_size_cap_stops_writing_with_one_warning(tmp_path, caplog):
    tr = Tracer(tmp_path / "t", max_file_mb=0.0005)  # ~524 bytes
    with caplog.at_level(logging.WARNING, logger="citerag.tracing"):
        for i in range(10):
            tr.write({"i": i, "pad": "x" * 200})
    rows = read_traces(tr.directory)
    assert 1 < len(rows) < 10 and [r["i"] for r in rows] == list(range(len(rows)))
    assert caplog.text.count("not writing more traces today") == 1


def test_retention_deletes_only_old_day_files(tmp_path):
    d = tmp_path / "t"
    d.mkdir()
    today = datetime.now(timezone.utc).date()
    names = {
        "old": f"{today - timedelta(days=15)}.jsonl", "older": "2020-01-01.jsonl",
        "edge": f"{today - timedelta(days=14)}.jsonl", "recent": f"{today - timedelta(days=1)}.jsonl",
        "notes": "notes.jsonl", "invalid": "2020-13-45.jsonl", "other": "2020-01-01.txt",
    }
    for name in names.values():
        (d / name).write_text("{}\n")
    tr = Tracer(d, retention_days=14)
    tr.write({"n": 1})
    assert sorted(p.name for p in d.iterdir()) == sorted(
        [names["edge"], names["recent"], names["notes"], names["invalid"], names["other"], f"{today}.jsonl"])


def test_retention_errors_are_logged_not_raised(tmp_path, monkeypatch, caplog):
    d = tmp_path / "t"
    d.mkdir()
    (d / "2020-01-01.jsonl").write_text("{}\n")

    def refuse(self, *a, **kw):
        raise PermissionError("nope")

    monkeypatch.setattr(Path, "unlink", refuse)
    with caplog.at_level(logging.WARNING, logger="citerag.tracing"):
        Tracer(d).write({"n": 1})
    assert "could not prune" in caplog.text and read_traces(d)[-1] == {"n": 1}


def test_metrics_limit_and_cache(pipeline, tracer, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(api_module, "_now", lambda: clock[0])
    client = TestClient(create_app(pipeline, tracer))
    assert client.get("/metrics", params={"last": 1001}).status_code == 422
    assert client.get("/metrics", params={"last": 1000}).status_code == 200
    tracer.write(_t(total=1))
    first = client.get("/metrics").json()
    assert first["requests"] == 1 and first["last"] == 200
    tracer.write(_t(total=2))
    clock[0] += 4.9
    assert client.get("/metrics").json() == first  # cached
    assert client.get("/metrics", params={"last": 5}).json()["requests"] == 2  # other `last`: own entry
    clock[0] += 0.2
    assert client.get("/metrics").json()["requests"] == 2  # expired


def test_api_without_tracer(pipeline):
    client = TestClient(create_app(pipeline))
    m = client.get("/metrics").json()
    assert m["enabled"] is False and m["requests"] == 0
    assert client.get("/api/health").json()["tracing"] is False


def test_settings_load_tracing_and_pricing(tmp_path):
    (tmp_path / "citerag.toml").write_text(
        '[tracing]\nenabled = false\ndir = "tr"\n[pricing]\n"m1" = { input = 1.5, output = 3 }\n')
    s = Settings.load(tmp_path / "citerag.toml")
    assert s.tracing_enabled is False and s.trace_path == (tmp_path / "tr").resolve()
    assert s.pricing == {"m1": {"input": 1.5, "output": 3}}
    d = Settings()
    assert d.tracing_enabled is True and d.trace_dir == "traces" and d.pricing == {}


def test_repo_toml_has_pricing():
    s = Settings.load(Path(__file__).resolve().parent.parent / "citerag.toml")
    assert s.pricing["claude-opus-5-5"] == {"input": 4.0, "output": 20.0}
    assert s.tracing_enabled is True and s.trace_max_file_mb == 50 and s.trace_retention_days == 14
    assert s.pricing["claude-opus-5"] == {"input": 5.0, "output": 25.0}
    assert s.pricing["claude-sonnet-5"] == {"input": 2.0, "output": 10.0}
    assert s.pricing["claude-opus-4-8"] == {"input": 5.0, "output": 25.0}


def test_cli_traces(tmp_path, capsys):
    cfg = tmp_path / "citerag.toml"
    cfg.write_text("")
    assert main(["--config", str(cfg), "traces"]) == 0
    assert capsys.readouterr().out.startswith("no traces in ")
    tr = Tracer(tmp_path / "traces")
    tr.write(_t(cites=2, retrieve=10, gate=1, generate=100, total=111))
    tr.write(_t(kind="search", gen=None, retrieve=5, total=5))
    assert main(["--config", str(cfg), "traces", "--last", "5"]) == 0
    out = capsys.readouterr().out
    assert "last 2 requests" in out and "1 asks, 1 searches" in out and "by generator: ollama 1" in out
    assert main(["--config", str(cfg), "traces", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["requests"] == 2 and data["malformed_lines"] == 0


def test_env_overrides_tracing_enabled(tmp_path, monkeypatch):
    cfg = tmp_path / "citerag.toml"
    cfg.write_text("[tracing]\nenabled = true\nmax_file_mb = 1.5\nretention_days = 3\n")
    monkeypatch.delenv("CITERAG_TRACING", raising=False)
    s = Settings.load(cfg)
    assert s.tracing_enabled is True and s.trace_max_file_mb == 1.5 and s.trace_retention_days == 3
    monkeypatch.setenv("CITERAG_TRACING", "0")
    assert Settings.load(cfg).tracing_enabled is False
    cfg.write_text("[tracing]\nenabled = false\n")
    monkeypatch.setenv("CITERAG_TRACING", "1")
    assert Settings.load(cfg).tracing_enabled is True
    monkeypatch.setenv("CITERAG_TRACING", "maybe")
    assert Settings.load(cfg).tracing_enabled is False  # unrecognised values leave the file in charge


@pytest.mark.parametrize("value", ["0", "-3", "abc", "1.5"])
def test_cli_traces_last_must_be_positive_int(tmp_path, capsys, value):
    cfg = tmp_path / "citerag.toml"
    cfg.write_text("")
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(cfg), "traces", "--last", value])
    assert exc.value.code == 2 and "--last" in capsys.readouterr().err


# Not covered: a real HTTP client disconnect. Starlette's TestClient runs the app to completion before returning, so
# the cancel path is tested at the traced_stream level (test_client_disconnect) instead.
