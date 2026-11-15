from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import pytest

from citerag import cli, mcp_server
from citerag.generation import ClaudeGenerator, ExtractiveGenerator, OllamaGenerator
from citerag.mcp_server import CiteragTools
from citerag.tracing import Tracer, format_summary, load_traces, summarise, trace_chunk_read

QUESTION = "How are partial days billed for labor days?"


@pytest.fixture
def tracer(tmp_path):
    return Tracer(tmp_path / "traces")


@pytest.fixture
def tools(pipeline, corpus_settings, tracer):
    return CiteragTools(pipeline, corpus_settings, tracer)


def _traces(tracer):
    return load_traces(tracer.directory)[0]


def _code(result):
    return result["error"]["code"]


# ---------- search_sources ----------


def test_search_sources_shape(tools, tracer):
    r = tools.search_sources("labor days partial", k=3)
    json.dumps(r)
    assert r["mode"] == "hybrid" and r["query"] == "labor days partial"
    assert 1 <= len(r["results"]) <= 3
    for rank, item in enumerate(r["results"], 1):
        assert item["rank"] == rank
        assert set(item) == {"rank", "file", "start_line", "end_line", "heading", "score", "text"}
        assert item["heading"] and item["text"] and item["start_line"] <= item["end_line"]
    assert any("Labor" in i["heading"] for i in r["results"])
    t = _traces(tracer)
    assert [(x["kind"], x["source"]) for x in t] == [("search", "mcp")]


@pytest.mark.parametrize("kwargs", [
    {"query": ""}, {"query": "   "}, {"query": "x" * 1001}, {"query": "ok", "k": 0}, {"query": "ok", "k": 21},
    {"query": "ok", "k": True}, {"query": "ok", "mode": "fuzzy"},
])
def test_search_sources_validation(tools, kwargs):
    assert _code(tools.search_sources(**kwargs)) == "invalid_argument"


def test_search_rejections_are_traced(tools, tracer):
    tools.search_sources("", k=3)
    tools.search_sources("ok", k=99)
    tools.search_sources("ok", mode="fuzzy")
    t = _traces(tracer)
    assert [(x["kind"], x["status"], x["error"]["type"]) for x in t] == [("search", "error", "invalid_argument")] * 3


def test_search_sources_internal_error(tools, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret /path")

    monkeypatch.setattr(tools.pipeline.retriever, "search", boom)
    r = tools.search_sources("labor")
    assert _code(r) == "internal" and "RuntimeError" in r["error"]["message"] and "secret" not in str(r)


# ---------- get_chunk ----------


def test_get_chunk_exact_lines(tools, corpus_settings):
    r = tools.get_chunk("proj/calc.js", 4, 6)
    on_disk = (corpus_settings.root / "corpus/proj/calc.js").read_text().split("\n")
    assert r["text"] == "\n".join(on_disk[3:6])
    assert (r["start_line"], r["end_line"], r["stale"]) == (4, 6, False)
    assert r["total_lines"] == len(on_disk) - 1


def test_get_chunk_html_line_range_matches_search_hit(tools):
    hits = tools.search_sources("Machine Learning Models KNN CNN", k=5)["results"]
    hit = next(h for h in hits if h["file"] == "site/index.html")
    r = tools.get_chunk("site/index.html", hit["start_line"], hit["end_line"])
    assert r["text"] == hit["text"]
    assert "<h2>" not in r["text"] and "Machine Learning Models" in r["text"]


def test_get_chunk_covers_every_line_of_a_multi_paragraph_hit(tools):
    hits = tools.search_sources("labor days partial billed deployment", k=5)["results"]
    hit = max((h for h in hits if h["file"] == "proj/README.md"), key=lambda h: h["end_line"] - h["start_line"])
    assert hit["end_line"] > hit["start_line"]
    text = tools.get_chunk(hit["file"], hit["start_line"], hit["end_line"])["text"]
    lines = [ln.strip() for ln in hit["text"].split("\n") if ln.strip()]
    assert len(lines) > 2
    for ln in lines:
        assert ln in text


def test_get_chunk_clamps_and_limits(tools):
    total = tools.get_chunk("proj/calc.js", 1, 1)["total_lines"]
    r = tools.get_chunk("proj/calc.js", total, total + 50)
    assert r["end_line"] == total
    assert _code(tools.get_chunk("proj/calc.js", 1, 401)) == "invalid_argument"
    assert tools.get_chunk("proj/calc.js", 1, 400)["end_line"] == total
    assert _code(tools.get_chunk("proj/calc.js", total + 1, total + 2)) == "invalid_argument"
    assert _code(tools.get_chunk("proj/calc.js", 0, 3)) == "invalid_argument"
    assert _code(tools.get_chunk("proj/calc.js", 5, 3)) == "invalid_argument"
    assert _code(tools.get_chunk("proj/calc.js", "1", 3)) == "invalid_argument"


def test_get_chunk_stale_and_deleted(tools, corpus_settings):
    p = corpus_settings.root / "corpus/proj/calc.js"
    p.write_text(p.read_text() + "// more\n")
    assert tools.get_chunk("proj/calc.js", 1, 2)["stale"] is True
    p.unlink()
    assert _code(tools.get_chunk("proj/calc.js", 1, 2)) == "not_found"


def test_get_chunk_trace(tools, tracer):
    tools.get_chunk("proj/calc.js", 1, 3)
    tools.get_chunk("../x", 1, 3)
    ok, bad = _traces(tracer)
    assert (ok["kind"], ok["source"], ok["status"], ok["cost_usd"]) == ("chunk", "mcp", "ok", 0.0)
    assert ok["question"] == "proj/calc.js:1-3"
    assert ok["hits"][0]["file"] == "proj/calc.js" and ok["latency_ms"]["total"] is not None
    assert bad["status"] == "error"


@pytest.mark.parametrize("file", [
    "../secrets.txt", "proj/../../etc/passwd", "/etc/passwd", "~/x", "proj\\calc.js", "proj/calc.js\x00", "",
    "proj/node_modules/lib/index.js", "proj/notindexed.md", "proj//calc.js", "proj/./calc.js", "PROJ/calc.js",
])
def test_get_chunk_rejects_outside_index(tools, corpus_settings, monkeypatch, file):
    (corpus_settings.root / "corpus/proj/notindexed.md").write_text("# secret\n")
    opened = []
    real_read = type(corpus_settings.root).read_bytes
    monkeypatch.setattr(type(corpus_settings.root), "read_bytes", lambda self: opened.append(self) or real_read(self))
    r = tools.get_chunk(file, 1, 2)
    assert _code(r) in ("not_indexed", "invalid_argument")
    assert opened == []


def test_get_chunk_rejects_symlink_escape(tools, corpus_settings, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("top secret\n")
    link = corpus_settings.root / "corpus/proj/link.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not supported")
    tools.pipeline.retriever.index.files = {**tools.pipeline.retriever.index.files, "proj/link.md": "0" * 64}
    r = tools.get_chunk("proj/link.md", 1, 1)
    assert _code(r) == "not_indexed" and "secret" not in str(r)


def _add_indexed(tools, rel, content: bytes, folder="proj"):
    """Write a file into a source and register it in the index, as if the indexer had seen it."""
    path = tools.settings.root / "corpus" / folder / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    index = tools.pipeline.retriever.index
    index.files = {**index.files, f"{folder}/{rel}": hashlib.sha256(content).hexdigest()}
    return path


def test_get_chunk_too_large_and_binary(tools, tracer, monkeypatch):
    _add_indexed(tools, "big.txt", b"x\n" * 600_000)
    _add_indexed(tools, "blob.bin", b"abc\x00def\n")
    reads = []
    real = type(tools.settings.root).read_bytes
    monkeypatch.setattr(type(tools.settings.root), "read_bytes", lambda self: reads.append(self) or real(self))
    assert _code(tools.get_chunk("proj/big.txt", 1, 2)) == "too_large"
    assert reads == []  # size is checked from stat before reading
    assert _code(tools.get_chunk("proj/blob.bin", 1, 2)) == "not_text"
    assert [x["error"]["type"] for x in _traces(tracer)] == ["too_large", "not_text"]


def test_get_chunk_truncates_long_text(tools):
    _add_indexed(tools, "min.js", b"var a=" + b"1+" * 30_000 + b"1;\n")
    r = tools.get_chunk("proj/min.js", 1, 1)
    assert r["truncated"] is True and len(r["text"]) == mcp_server.MAX_TEXT_CHARS
    assert tools.get_chunk("proj/calc.js", 1, 3)["truncated"] is False


def test_get_chunk_internal_error_is_traced(tools, tracer, monkeypatch):
    def boom(self, file, start, end):
        raise RuntimeError("secret")

    monkeypatch.setattr(CiteragTools, "_get_chunk", boom)
    r = tools.get_chunk("proj/calc.js", 1, 2)
    assert _code(r) == "internal" and "secret" not in str(r)
    (t,) = _traces(tracer)
    assert (t["kind"], t["status"], t["error"]["type"]) == ("chunk", "error", "internal")


def test_get_chunk_error_codes_in_traces(tools, tracer, corpus_settings):
    tools.get_chunk("../x", 1, 2)
    tools.get_chunk("proj/calc.js", 0, 2)
    (corpus_settings.root / "corpus/proj/calc.js").unlink()
    tools.get_chunk("proj/calc.js", 1, 2)
    assert [x["error"]["type"] for x in _traces(tracer)] == ["not_indexed", "invalid_argument", "not_found"]


@pytest.mark.parametrize("injected", ["../outside.md", "proj/../../outside.md", "proj/sub/../calc.js"])
def test_get_chunk_rejects_injected_index_ids_without_reading(tools, corpus_settings, monkeypatch, injected):
    """Even if a hostile index lists a traversal id, it is refused before any file is read."""
    (corpus_settings.root / "outside.md").write_text("top secret\n")
    (corpus_settings.root / "corpus/proj/sub").mkdir()
    index = tools.pipeline.retriever.index
    monkeypatch.setattr(index, "files", {**index.files, injected: "0" * 64})
    reads = []
    real = type(corpus_settings.root).read_bytes
    monkeypatch.setattr(type(corpus_settings.root), "read_bytes", lambda self: reads.append(self) or real(self))
    r = tools.get_chunk(injected, 1, 1)
    assert _code(r) == "not_indexed" and "secret" not in str(r)
    assert reads == []


def test_get_chunk_rejects_symlinked_directory_escape(tools, corpus_settings, tmp_path, monkeypatch):
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (outside / "x.md").write_text("top secret\n")
    try:
        (corpus_settings.root / "corpus/proj/linkdir").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not supported")
    index = tools.pipeline.retriever.index
    monkeypatch.setattr(index, "files", {**index.files, "proj/linkdir/x.md": "0" * 64})
    reads = []
    real = type(tmp_path).read_bytes
    monkeypatch.setattr(type(tmp_path), "read_bytes", lambda self: reads.append(self) or real(self))
    r = tools.get_chunk("proj/linkdir/x.md", 1, 1)
    assert _code(r) == "not_indexed" and reads == []


def test_get_chunk_rejects_file_symlinked_inside_source(tools, corpus_settings):
    link = corpus_settings.root / "corpus/proj/alias.js"
    try:
        link.symlink_to(corpus_settings.root / "corpus/proj/calc.js")
    except OSError:
        pytest.skip("symlinks not supported")
    index = tools.pipeline.retriever.index
    index.files = {**index.files, "proj/alias.js": index.files["proj/calc.js"]}
    assert _code(tools.get_chunk("proj/alias.js", 1, 2)) == "not_indexed"


def test_get_chunk_duplicate_source_names_use_index_hash(pipeline, corpus_settings, tracer):
    root = corpus_settings.root
    (root / "a/proj").mkdir(parents=True)
    (root / "b/proj").mkdir(parents=True)
    (root / "a/proj/f.md").write_text("from a\n")
    (root / "b/proj/f.md").write_text("from b\n")
    settings = dataclasses.replace(corpus_settings, sources=["a/proj", "b/proj"])
    index = pipeline.retriever.index
    index.files = {"proj/f.md": hashlib.sha256(b"from b\n").hexdigest(), "proj/g.md": "0" * 64}
    (root / "a/proj/g.md").write_text("only a\n")
    (root / "b/proj/g.md").write_text("only b\n")
    t = CiteragTools(pipeline, settings, tracer)
    r = t.get_chunk("proj/f.md", 1, 1)
    assert r["text"] == "from b" and r["stale"] is False
    g = t.get_chunk("proj/g.md", 1, 1)  # no candidate matches the hash: first source, flagged stale
    assert g["text"] == "only a" and g["stale"] is True


def test_get_chunk_single_file_source(pipeline, corpus_settings, tracer):
    root = corpus_settings.root
    (root / "notes.md").write_text("# one\nline\n")
    settings = dataclasses.replace(corpus_settings, sources=["notes.md"])
    pipeline.retriever.index.files = {"notes.md": hashlib.sha256(b"# one\nline\n").hexdigest()}
    r = CiteragTools(pipeline, settings, tracer).get_chunk("notes.md", 1, 2)
    assert r["text"] == "# one\nline" and r["stale"] is False


def test_ingest_warns_on_duplicate_source_names(corpus_settings, embedder, capsys):
    from citerag.engine import ingest

    root = corpus_settings.root
    for d in ("a/proj", "b/proj"):
        (root / d).mkdir(parents=True)
        (root / d / "f.md").write_text(f"# {d}\n")
    ingest(dataclasses.replace(corpus_settings, sources=["a/proj", "b/proj"]), embedder)
    assert "share the name 'proj'" in capsys.readouterr().err


def test_get_chunk_non_string_file(tools):
    assert _code(tools.get_chunk(None, 1, 2)) == "invalid_argument"


# ---------- ask ----------


class FakeGenerator:
    name = "fake"

    def __init__(self, events):
        self.events = events

    def stream(self, question, hits):
        for ev in self.events:
            if isinstance(ev, Exception):
                raise ev
            yield ev


def test_ask_inline_markers_follow_citation_order(tools, tracer):
    tools.pipeline.generator = FakeGenerator([
        {"type": "token", "text": "Partial days are billed whole."}, {"type": "cite", "hits": [2]},
        {"type": "token", "text": " VAT is separate."}, {"type": "cite", "hits": [0, 2]},
        {"type": "done", "generator": "fake", "model": "m"},
    ])
    r = tools.ask(QUESTION)
    json.dumps(r)
    assert r["answer"] == "Partial days are billed whole.[1] VAT is separate.[2][1]"
    assert r["generator"] == "fake" and r["model"] == "m" and r["abstained"] is False
    assert [c["n"] for c in r["citations"]] == [1, 2]
    assert set(r["citations"][0]) == {"n", "file", "start_line", "end_line", "heading"}
    for n in map(int, re.findall(r"\[(\d+)\]", r["answer"])):
        assert r["citations"][n - 1]["n"] == n
    assert r["segments"] == [{"text": "Partial days are billed whole.", "cites": [1]},
                             {"text": " VAT is separate.", "cites": [2, 1]}]
    assert "get_chunk" in r["sources_note"]
    assert [(t["kind"], t["source"]) for t in _traces(tracer)] == [("ask", "mcp")]


def test_ask_extractive(tools):
    r = tools.ask(QUESTION)
    assert r["abstained"] is False and r["generator"] == "extractive" and r["citations"]
    assert r["segments"] and all(seg["cites"] for seg in r["segments"][:-1])  # literal [n] in text is not trusted
    for seg in r["segments"]:
        for n in seg["cites"]:
            assert r["citations"][n - 1]["file"]


def test_ask_abstains(tools):
    r = tools.ask("What is the airspeed velocity of an unladen swallow?")
    assert r["abstained"] is True and r["citations"] == [] and r["generator"] == "abstain"


def test_ask_generator_error_event(tools):
    tools.pipeline.generator = FakeGenerator([
        {"type": "token", "text": "x"}, {"type": "error", "message": "Could not reach Ollama."}])
    r = tools.ask(QUESTION)
    assert r == {"error": {"code": "generator_error", "message": "Could not reach Ollama."}}


def test_ask_generator_raises(tools, tracer):
    tools.pipeline.generator = FakeGenerator([ValueError("boom /secret")])
    r = tools.ask(QUESTION)
    assert _code(r) == "internal" and "ValueError" in r["error"]["message"] and "secret" not in str(r)
    assert _traces(tracer)[0]["status"] == "error"


def test_ask_validation(tools, tracer):
    assert _code(tools.ask("  ")) == "invalid_argument"
    assert _code(tools.ask("x" * 1001)) == "invalid_argument"
    t = _traces(tracer)
    assert [(x["kind"], x["status"], x["error"]["type"]) for x in t] == [("ask", "error", "invalid_argument")] * 2


def test_ask_keeps_literal_brackets_out_of_segments_cites(tools):
    tools.pipeline.generator = FakeGenerator([
        {"type": "token", "text": "Use arr[1] here."}, {"type": "cite", "hits": [0]},
        {"type": "done", "generator": "fake", "model": "m"}])
    r = tools.ask(QUESTION)
    assert r["segments"] == [{"text": "Use arr[1] here.", "cites": [1]}]


# ---------- tracing ----------


def test_chunk_traces_counted_separately(tracer):
    base = {"kind": "search", "source": "mcp", "status": "ok", "latency_ms": {"total": 10.0}}
    traces = [base, {**base, "kind": "chunk", "latency_ms": {"total": 9999.0}, "cost_usd": 0.0}]
    s = summarise(traces)
    assert (s["chunks"], s["searches"], s["asks"], s["requests"]) == (1, 1, 0, 2)
    assert s["search_latency_ms"]["n"] == 1 and s["search_latency_ms"]["p95"] == 10.0
    assert s["latency_ms"]["total"]["n"] == 0
    assert "1 chunks" in format_summary(s)
    trace_chunk_read(tracer, "a/b.py", 1, 5, 2.0, ok=True)
    assert summarise(_traces(tracer))["chunks"] == 1


def test_chunk_errors_do_not_inflate_error_rate(tracer):
    base = {"source": "mcp", "latency_ms": {"total": 1.0}}
    traces = [{**base, "kind": "ask", "status": "ok"}, {**base, "kind": "search", "status": "error",
                                                         "error": {"type": "ValueError"}}]
    trace_chunk_read(tracer, "a.py", 1, 2, 1.0, ok=False, code="too_large")
    trace_chunk_read(tracer, "a.py", 1, 2, 1.0, ok=False, code="not_text")
    trace_chunk_read(tracer, "a.py", 1, 2, 1.0, ok=False, code="not_text")
    s = summarise(traces + _traces(tracer))
    assert s["error_rate"] == 0.5 and s["errors_by_type"] == {"ValueError": 1}
    assert s["chunk_errors_by_type"] == {"too_large": 1, "not_text": 2}
    text = format_summary(s)
    assert "error rate 50.0%" in text and "too_large 1" in text and "not_text 2" in text


# ---------- SDK ----------


def _client_run(tools, fn):
    pytest.importorskip("mcp")
    import anyio
    from mcp import Client

    server = mcp_server.build_server(tools)

    async def go():
        async with Client(server) as c:
            return await fn(c)

    return anyio.run(go)


def test_sdk_list_tools(tools):
    async def fn(c):
        return await c.list_tools()

    listed = _client_run(tools, fn)
    by_name = {t.name: t for t in listed.tools}
    assert set(by_name) == {"search_sources", "get_chunk", "ask"}
    for t in by_name.values():
        assert t.description and t.annotations.read_only_hint is True and t.annotations.open_world_hint is False
    assert by_name["ask"].annotations.idempotent_hint is False


def test_sdk_calls(tools, capsys):
    async def fn(c):
        s = await c.call_tool("search_sources", {"query": "labor days", "k": 2})
        g = await c.call_tool("get_chunk", {"file": "../x", "start_line": 1, "end_line": 2})
        a = await c.call_tool("ask", {"question": QUESTION})
        return s, g, a

    s, g, a = _client_run(tools, fn)
    print(repr(s), file=sys.stderr)
    assert not s.is_error and 1 <= len(s.structured_content["results"]) <= 2
    # A tool error is data (structured, readable) but flagged so the client model knows the call failed.
    assert g.is_error and g.structured_content["error"]["code"] == "not_indexed"
    assert json.loads(g.content[0].text) == g.structured_content
    assert not a.is_error and a.structured_content["answer"] and a.structured_content["citations"]


def test_sdk_schema_has_mode_enum_and_invalid_arguments_are_errors(tools, tracer):
    async def fn(c):
        listed = await c.list_tools()
        a = await c.call_tool("get_chunk", {"file": "proj/calc.js", "start_line": 1})
        b = await c.call_tool("search_sources", {"query": "labor", "k": "abc"})
        d = await c.call_tool("search_sources", {"query": "labor", "mode": "fuzzy"})
        return listed, a, b, d

    listed, a, b, d = _client_run(tools, fn)
    schema = next(t for t in listed.tools if t.name == "search_sources").input_schema
    assert schema["properties"]["mode"]["enum"] == ["hybrid", "bm25", "dense"]
    assert a.is_error is True and b.is_error is True and d.is_error is True
    t = _traces(tracer)
    assert [(x["kind"], x["source"], x["status"], x["error"]["type"]) for x in t] == [
        ("chunk", "mcp", "error", "invalid_arguments"), ("search", "mcp", "error", "invalid_arguments"),
        ("search", "mcp", "error", "invalid_arguments")]


def test_sdk_structured_errors_are_not_traced_twice(tools, tracer):
    async def fn(c):
        return await c.call_tool("get_chunk", {"file": "../x", "start_line": 1, "end_line": 2})

    assert _client_run(tools, fn).is_error is True
    assert [x["error"]["type"] for x in _traces(tracer)] == ["not_indexed"]


# ---------- generator choice ----------


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("CITERAG_GENERATOR", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return monkeypatch


def test_mcp_generator_ignores_api_key(clean_env, corpus_settings):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(mcp_server.mcp_generator(corpus_settings), ExtractiveGenerator)
    clean_env.setenv("CITERAG_GENERATOR", "  ")
    assert isinstance(mcp_server.mcp_generator(corpus_settings), ExtractiveGenerator)


def test_mcp_generator_env_claude(clean_env, corpus_settings):
    clean_env.setenv("CITERAG_GENERATOR", "claude")
    assert isinstance(mcp_server.mcp_generator(corpus_settings), ClaudeGenerator)


def test_mcp_generator_provider_ollama(clean_env, corpus_settings):
    gen = mcp_server.mcp_generator(dataclasses.replace(corpus_settings, provider="ollama"))
    assert isinstance(gen, OllamaGenerator)


def test_mcp_generator_unknown_choice_raises(clean_env, corpus_settings):
    with pytest.raises(ValueError):
        mcp_server.mcp_generator(dataclasses.replace(corpus_settings, provider="gpt"))


# ---------- stdio ----------


def test_stdio_server_speaks_clean_jsonrpc(tmp_path):
    pytest.importorskip("mcp")
    from conftest import write_corpus
    from citerag.config import Settings
    from citerag.embeddings import HashingEmbedder
    from citerag.engine import ingest

    write_corpus(tmp_path / "corpus")
    cfg = tmp_path / "citerag.toml"
    cfg.write_text('[corpus]\nsources = ["corpus/proj", "corpus/site"]\n'
                   '[index]\npath = "index"\nembedder = "hashing:256"\n')
    ingest(Settings.load(cfg), HashingEmbedder(256))
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "CITERAG_GENERATOR")}
    proc = subprocess.Popen([sys.executable, "-m", "citerag.cli", "--config", str(cfg), "mcp"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(target=lambda: [lines.put(ln) for ln in proc.stdout], daemon=True).start()
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    seen: list[dict] = []
    try:
        for r in requests:
            proc.stdin.write(json.dumps(r) + "\n")
        proc.stdin.flush()
        deadline = time.monotonic() + 15
        while not any(m.get("id") == 2 for m in seen) and time.monotonic() < deadline:
            try:
                seen.append(json.loads(lines.get(timeout=0.5)))  # any non-JSON line on stdout fails here
            except queue.Empty:
                pass
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert all(m.get("jsonrpc") == "2.0" for m in seen)
    listing = next(m for m in seen if m.get("id") == 2)
    assert {t["name"] for t in listing["result"]["tools"]} == {"search_sources", "get_chunk", "ask"}
    assert "ask uses extractive" in proc.stderr.read()


# ---------- CLI ----------


def test_cli_mcp_without_sdk(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)  # may already be imported by other tests
    assert cli.main(["mcp"]) == 1
    assert 'pip install -e ".[mcp]"' in capsys.readouterr().err


def test_cli_mcp_missing_index(tmp_path, capsys):
    pytest.importorskip("mcp")
    cfg = tmp_path / "citerag.toml"
    cfg.write_text('[index]\npath = "nope"\nembedder = "hashing:256"\n')
    assert cli.main(["--config", str(cfg), "mcp"]) == 1
    cap = capsys.readouterr()
    assert "no index" in cap.err and cap.out == ""
