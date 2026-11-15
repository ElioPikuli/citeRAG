from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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


def read_traces(directory: Path) -> list[dict]:
    return [json.loads(line) for f in sorted(directory.glob("*.jsonl")) for line in f.read_text().splitlines()]


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces", pricing=PRICING)


def test_successful_ask_trace(pipeline, tracer):
    q = "How are partial labor days billed? " + "labor " * 70
    expected = list(pipeline.stream(q))
    events = list(traced_stream(pipeline, tracer, q))
    assert [e["type"] for e in events] == [e["type"] for e in expected]
    assert events[0] == expected[0] or events[0]["hits"] == expected[0]["hits"]
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
    assert t["status"] == "error" and t["error"]["type"] == "client_disconnected"
    assert t["hits"] and t["latency_ms"]["total"] is not None


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
        _t(status="error", error={"type": "generator_error", "message": "m"}, cost=None, retrieve=50, gate=4, total=60),
    ]
    s = summarise(traces)
    assert (s["requests"], s["asks"], s["searches"]) == (5, 4, 1)
    assert s["abstain_rate"] == 0.25 and s["error_rate"] == 0.2
    assert s["latency_ms"]["retrieve"] == {"p50": 30, "p95": 50, "n": 5}
    assert s["latency_ms"]["gate"] == {"p50": 2, "p95": 4, "n": 4}
    assert s["latency_ms"]["generate"] == {"p50": 100, "p95": 200, "n": 2}
    assert s["latency_ms"]["total"] == {"p50": 60, "p95": 224, "n": 5}
    assert s["citations_per_answer"] == 3.0
    assert s["tokens"] == {"input": 150, "output": 15}
    assert s["cost_usd"] == {"total": 0.5, "unknown": 2}
    assert s["errors_by_type"] == {"generator_error": 1}
    assert s["by_source"] == {"api": 4, "cli": 1} and s["by_generator"] == {"ollama": 3, "abstain": 1}
    assert s["first_ts"] == "2026-01-01T00:00:00Z" and s["last_ts"] == "2026-01-01T00:00:03Z"
    assert "SECRET" not in json.dumps(s) and "SECRET" not in format_summary(s)


def test_summarise_empty():
    s = summarise([])
    assert s["requests"] == 0 and s["abstain_rate"] is None and s["error_rate"] is None
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
    r = client.post("/api/ask", json={"question": "How are partial labor days billed?"})
    assert r.status_code == 200
    assert client.get("/api/search", params={"q": "calcVat", "k": 2}).status_code == 200
    asks, searches = read_traces(tracer.directory)
    assert asks["kind"] == "ask" and asks["source"] == "api" and asks["cost_usd"] == round(2000 * 4e-6 + 100 * 20e-6, 8)
    assert searches["kind"] == "search" and searches["source"] == "api"
    m = client.get("/metrics", params={"last": 10}).json()
    assert m["enabled"] is True and m["requests"] == 2 and m["asks"] == 1 and m["malformed_lines"] == 0
    assert "question" not in json.dumps(m)
    assert client.get("/metrics", params={"last": 0}).status_code == 422


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
    assert s.tracing_enabled is True


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
