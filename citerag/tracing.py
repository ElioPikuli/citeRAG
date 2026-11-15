"""Request tracing: one JSON line per ask/search/chunk read in {trace_dir}/{YYYY-MM-DD}.jsonl.

Tracing must never break a request, so every write failure and every bookkeeping error is logged and swallowed;
only exceptions raised by the pipeline itself are recorded as request errors. Traces hold the question truncated to
300 characters and hit locations, never chunk text; `summarise` goes further and keeps only aggregates, because
/metrics is served publicly.

Latency note: an ask's `total` is wall-clock from the first pull to the last event, so it includes the time the
consumer spends between events (an SSE client reading slowly, a CLI printing). The stages (retrieve, gate, generate)
are measured inside the pipeline and therefore do not sum to total.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .retriever import Hit
from .stats import percentile

log = logging.getLogger("citerag.tracing")

QUESTION_MAX = 300
HITS_MAX = 10
FREE_GENERATORS = ("extractive", "abstain", "ollama")
STAGES = ("retrieve", "gate", "generate", "total")
TAIL_BYTES = 4 * 1024 * 1024  # load_traces reads at most this much from the end of each file


class Tracer:
    def __init__(self, directory: Path, *, enabled: bool = True, pricing: dict[str, dict] | None = None,
                 max_file_mb: float = 50, retention_days: int = 14):
        self.directory = Path(directory)
        self.enabled = enabled
        self.pricing = pricing or {}
        self.max_file_bytes = int(max_file_mb * 1024 * 1024)
        self.retention_days = retention_days
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a threadpool
        self._cleaned_day: str | None = None
        self._capped_day: str | None = None

    def write(self, trace: dict) -> None:
        """Append one line to today's file (UTC). Never raises.

        The file is public-server input, so it is bounded: past `max_file_mb` writing stops for the day (one
        warning), and the first write of each day deletes files older than `retention_days`."""
        try:
            line = json.dumps(trace, ensure_ascii=False)
            now = datetime.now(timezone.utc)
            day = f"{now:%Y-%m-%d}"
            path = self.directory / f"{day}.jsonl"
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                if self._cleaned_day != day:
                    self._cleaned_day = day
                    self._prune(now)
                if path.exists() and path.stat().st_size > self.max_file_bytes:
                    if self._capped_day != day:
                        self._capped_day = day
                        log.warning("trace file %s exceeds %s bytes; not writing more traces today",
                                    path.name, self.max_file_bytes)
                    return
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:  # noqa: BLE001 - tracing must never break a request
            log.warning("could not write trace: %s: %s", type(e).__name__, e)

    def _prune(self, now: datetime) -> None:
        """Delete day files older than the retention window. Errors are logged, never raised."""
        try:
            cutoff = (now - timedelta(days=self.retention_days)).date()
            for path in self.directory.glob("????-??-??.jsonl"):
                try:
                    if datetime.strptime(path.stem, "%Y-%m-%d").date() < cutoff:
                        path.unlink()
                except (ValueError, OSError) as e:
                    log.warning("could not prune %s: %s: %s", path.name, type(e).__name__, e)
        except Exception as e:  # noqa: BLE001
            log.warning("could not prune traces: %s: %s", type(e).__name__, e)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _price_for(model: str | None, pricing: dict) -> dict | None:
    """Exact key first, else the longest key that prefixes the model id at a '-' boundary, so a dated id
    ("claude-opus-5-5-20261101") prices as its family and "claude-opus-5" never shadows "claude-opus-5-5"."""
    if not model or not isinstance(pricing, dict):
        return None
    if model in pricing:
        price = pricing[model]
    else:
        keys = [k for k in pricing if isinstance(k, str) and model.startswith(k + "-")]
        if not keys:
            return None
        price = pricing[max(keys, key=len)]
    if isinstance(price, dict) and _num(price.get("input")) and _num(price.get("output")):
        return price
    return None


def estimate_cost(generator: str | None, model: str | None, usage: dict | None, pricing: dict) -> float | None:
    """USD estimate. Free generators cost 0.0; an unpriced model or unreported tokens give None, never 0."""
    if generator in FREE_GENERATORS:
        return 0.0
    price = _price_for(model, pricing)
    if price is None or not isinstance(usage, dict):
        return None
    tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
    if not _num(tokens_in) or not _num(tokens_out):
        return None
    return round(tokens_in * price["input"] / 1e6 + tokens_out * price["output"] / 1e6, 8)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 4)


def _hit_row(file: str, start: int, end: int, score, bm25_rank, dense_rank, dense) -> dict:
    return {"file": file, "start_line": start, "end_line": end, "score": _r(score),
            "bm25_rank": bm25_rank, "dense_rank": dense_rank, "dense": _r(dense)}


def _rows_from_dicts(hits: list[dict]) -> list[dict]:
    return [_hit_row(h.get("source"), h.get("start_line"), h.get("end_line"), h.get("score"),
                     h.get("bm25_rank"), h.get("dense_rank"), h.get("dense")) for h in hits[:HITS_MAX]]


def _rows_from_hits(hits: list[Hit]) -> list[dict]:
    return [_hit_row(h.chunk.source, h.chunk.start_line, h.chunk.end_line, h.score, h.bm25_rank,
                     h.dense_rank, h.dense) for h in hits[:HITS_MAX]]


def _blank(kind: str, source: str, question: str, k, mode) -> dict:
    return {
        "trace_id": uuid.uuid4().hex, "ts": _now(), "source": source, "kind": kind,
        "question": question[:QUESTION_MAX], "question_chars": len(question), "mode": mode, "k": k,
        "hits": [], "gate": None, "status": "ok", "generator": None, "model": None, "usage": None,
        "latency_ms": {s: None for s in STAGES}, "cost_usd": None, "citations": 0, "cite_events": 0,
        "markers": None, "invalid_markers": None, "stop_reason": None, "error": None,
    }


def _observe(trace: dict, ev: dict, state: dict, mode) -> None:
    """Fold one pipeline event into the trace. Pure bookkeeping: callers guard it."""
    kind = ev.get("type")
    if kind == "sources":
        rows = _rows_from_dicts(ev.get("hits", []))
        gate = ev.get("gate") or {}
        trace["hits"] = rows
        trace["gate"] = {"answerable": gate.get("answerable"), "coverage": gate.get("coverage"),
                         "max_dense": gate.get("max_dense")}
        trace["mode"] = gate.get("mode", mode)
        trace["latency_ms"]["retrieve"] = gate.get("retrieval_ms")
        trace["latency_ms"]["gate"] = gate.get("gate_ms")
    elif kind == "cite":
        trace["cite_events"] += 1
        state["cited"].update(ev.get("hits", []))
    elif kind == "done":
        state["done"] = ev
        state["ended"] = True
    elif kind == "error":
        trace["error"] = {"type": "generator_error", "message": str(ev.get("message"))[:300]}
        state["ended"] = True


def _finish_ask(trace: dict, state: dict, pipeline, tracer: Tracer, total_ms: float) -> None:
    done = state["done"]
    trace["citations"] = len(state["cited"])
    trace["latency_ms"]["total"] = round(total_ms, 1)
    if done is not None:
        trace["generator"] = done.get("generator")
        trace["model"] = done.get("model")
        trace["stop_reason"] = done.get("stop_reason")
        trace["markers"] = done.get("markers")
        trace["invalid_markers"] = done.get("invalid_markers")
        trace["latency_ms"]["generate"] = done.get("generation_ms")
        usage = done.get("usage")
        if isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage:
            trace["usage"] = {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
        if done.get("abstained"):
            trace["status"] = "abstained"
    else:
        # Failed before done: the configured generator is still known, which decides free vs unknown cost.
        trace["generator"] = getattr(getattr(pipeline, "generator", None), "name", None)
    if trace["error"] is not None:
        trace["status"] = "error"
    elif state["cancelled"]:
        trace["status"] = "cancelled"
    trace["cost_usd"] = estimate_cost(trace["generator"], trace["model"], trace["usage"], tracer.pricing)


def _flush_ask(trace: dict, state: dict, pipeline, tracer: Tracer, t0: float) -> None:
    """Build and write the trace once. Guarded: a bookkeeping failure is a warning, not a request failure."""
    if state["written"]:
        return
    state["written"] = True
    try:
        _finish_ask(trace, state, pipeline, tracer, (time.perf_counter() - t0) * 1000)
        tracer.write(trace)
    except Exception as e:  # noqa: BLE001 - never break the request
        log.warning("could not build trace: %s: %s", type(e).__name__, e)


def traced_stream(pipeline, tracer: Tracer | None, question: str, k: int | None = None, mode: str | None = None,
                  *, source: str = "api") -> Iterator[dict]:
    """Pass every event of pipeline.stream through unchanged and write one trace per stream.

    The trace is written when the `done` or `error` event arrives, before it is yielded, so it exists by the time
    the client sees the end of the answer. Streams that end any other way are written from `finally`: a consumer
    that closes early is "cancelled", a pipeline exception is an error of that type, and a generator that simply
    returns is an "incomplete" error."""
    if tracer is None or not tracer.enabled:
        yield from pipeline.stream(question, k, mode)
        return
    try:
        trace = _blank("ask", source, question, k or getattr(pipeline, "top_k", None), mode)
    except Exception as e:  # noqa: BLE001
        log.warning("could not start trace: %s: %s", type(e).__name__, e)
        yield from pipeline.stream(question, k, mode)
        return
    state: dict = {"cited": set(), "done": None, "ended": False, "cancelled": False, "written": False}
    t0 = time.perf_counter()
    inner = pipeline.stream(question, k, mode)
    raised = False
    try:
        for ev in inner:
            try:
                _observe(trace, ev, state, mode)
                if state["ended"]:
                    _flush_ask(trace, state, pipeline, tracer, t0)
            except Exception as e:  # noqa: BLE001 - bookkeeping must not break the stream
                log.warning("trace bookkeeping failed: %s: %s", type(e).__name__, e)
            yield ev
    except GeneratorExit:
        state["cancelled"] = not state["ended"]
        raise
    except BaseException as e:  # includes KeyboardInterrupt: record it, then let it propagate
        raised = True
        if not state["written"]:
            trace["error"] = {"type": type(e).__name__, "message": str(e)[:300]}
        raise
    finally:
        try:
            inner.close()
        except Exception:  # noqa: BLE001
            pass
        if not state["written"]:
            if not (raised or state["cancelled"] or state["ended"]):
                trace["error"] = {"type": "incomplete", "message": "stream ended without done or error"}
            _flush_ask(trace, state, pipeline, tracer, t0)


def traced_search(pipeline, tracer: Tracer | None, query: str, k: int, mode: str | None = None,
                  *, source: str = "api") -> list[Hit]:
    """pipeline.retriever.search with a trace. Pipeline exceptions are recorded and re-raised."""
    if tracer is None or not tracer.enabled:
        return pipeline.retriever.search(query, k, mode=mode)
    try:
        trace = _blank("search", source, query, k, mode or pipeline.retriever.mode)
        trace["cost_usd"] = 0.0  # retrieval only: local embeddings and BM25, no model call
    except Exception as e:  # noqa: BLE001
        log.warning("could not start trace: %s: %s", type(e).__name__, e)
        return pipeline.retriever.search(query, k, mode=mode)
    t0 = time.perf_counter()
    hits: list[Hit] | None = None
    try:
        hits = pipeline.retriever.search(query, k, mode=mode)
        return hits
    except BaseException as e:
        trace["error"] = {"type": type(e).__name__, "message": str(e)[:300]}
        trace["status"] = "error"
        raise
    finally:
        try:
            elapsed = round((time.perf_counter() - t0) * 1000, 1)
            trace["latency_ms"]["retrieve"] = elapsed
            trace["latency_ms"]["total"] = elapsed
            if hits is not None:
                trace["hits"] = _rows_from_hits(hits)
            tracer.write(trace)
        except Exception as e:  # noqa: BLE001 - never break the request
            log.warning("could not build trace: %s: %s", type(e).__name__, e)


def trace_chunk_read(tracer: Tracer | None, file: str, start: int, end: int, latency_ms: float, *, ok: bool,
                     code: str | None = None, source: str = "mcp") -> None:
    """One trace for a get_chunk call. No pipeline runs, so cost is 0.0 and only the location is recorded.
    A failed call records its error code (not_indexed, not_found, too_large, ...) as error.type."""
    if tracer is None or not tracer.enabled:
        return
    try:
        trace = _blank("chunk", source, f"{file}:{start}-{end}", None, None)
        trace["cost_usd"] = 0.0
        trace["hits"] = [_hit_row(file, start, end, None, None, None, None)]
        trace["latency_ms"]["total"] = round(latency_ms, 1)
        if not ok:
            trace["status"] = "error"
            trace["error"] = {"type": code or "chunk_error", "message": "get_chunk was rejected or failed"}
        tracer.write(trace)
    except Exception as e:  # noqa: BLE001 - never break the request
        log.warning("could not build trace: %s: %s", type(e).__name__, e)


def trace_rejected(tracer: Tracer | None, kind: str, question: str, code: str, *, k=None, mode=None,
                   source: str = "mcp") -> None:
    """Error trace for a call that was refused before any work happened (bad arguments)."""
    if tracer is None or not tracer.enabled:
        return
    try:
        trace = _blank(kind, source, question if isinstance(question, str) else "", k if _num(k) else None,
                       mode if isinstance(mode, str) else None)
        trace["cost_usd"] = 0.0
        trace["status"] = "error"
        trace["error"] = {"type": code, "message": "the call was rejected"}
        tracer.write(trace)
    except Exception as e:  # noqa: BLE001 - never break the request
        log.warning("could not build trace: %s: %s", type(e).__name__, e)


def _read_tail(path: Path, tail_bytes: int) -> list[str]:
    """Lines of the last `tail_bytes` of a file; a partial first line is dropped."""
    with path.open("rb") as f:
        size = f.seek(0, 2)
        start = max(0, size - tail_bytes - 1)  # one extra byte tells whether we landed on a line start
        f.seek(start)
        data = f.read()
    if start > 0:
        head, sep, rest = data.partition(b"\n")
        data = rest if sep else b""
    return data.decode("utf-8", errors="replace").split("\n")


def _str_or_none(x) -> bool:
    return x is None or isinstance(x, str)


def _valid(t: dict) -> bool:
    """Check the field types summarise relies on; absent fields are fine (old or foreign lines)."""
    lat = t.get("latency_ms")
    return (
        all(_str_or_none(t.get(f)) for f in ("status", "kind", "source", "generator", "ts"))
        and (t.get("citations") is None or (isinstance(t["citations"], int) and not isinstance(t["citations"], bool)))
        and (lat is None or (isinstance(lat, dict) and all(v is None or _num(v) for v in lat.values())))
        and (t.get("cost_usd") is None or _num(t["cost_usd"]))
        and all(t.get(f) is None or isinstance(t[f], dict) for f in ("error", "usage"))
    )


def load_traces(directory: Path, last: int = 200, *, tail_bytes: int = TAIL_BYTES) -> tuple[list[dict], int]:
    """Newest `last` traces (returned oldest first) and the number of malformed lines skipped on the way.

    Only the last `tail_bytes` of each file are read, newest file first, and reading stops once `last` traces
    are collected, so a huge or hostile trace file cannot make /metrics expensive."""
    directory = Path(directory)
    if not directory.is_dir():
        return [], 0
    out: list[dict] = []
    malformed = 0
    for path in sorted(directory.glob("????-??-??.jsonl"), reverse=True):
        if len(out) >= last:
            break
        try:
            lines = _read_tail(path, tail_bytes)
        except OSError:
            continue
        for line in reversed(lines):
            if len(out) >= last:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(obj, dict) and _valid(obj):
                out.append(obj)
            else:
                malformed += 1
    out.reverse()
    return out, malformed


def _stage(traces: list[dict], stage: str) -> dict:
    vals = [t["latency_ms"][stage] for t in traces
            if isinstance(t.get("latency_ms"), dict) and _num(t["latency_ms"].get(stage))]
    return {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "n": len(vals)}


def _sum_known(values: list) -> int:
    return sum(v for v in values if _num(v))


def summarise(traces: list[dict]) -> dict:
    """Aggregates only: no question text, no hit details.

    Populations are kept apart: error_rate counts status "error" over asks and searches (chunk reads are reported by
    chunk_errors_by_type instead), cancel_rate counts clients
    that left early, and the answer stage latencies cover only asks that finished (ok or abstained), so errors and
    cancellations don't drag the percentiles. Searches have their own latency."""
    asks = [t for t in traces if t.get("kind") == "ask"]
    searches = [t for t in traces if t.get("kind") == "search"]
    chunks = [t for t in traces if t.get("kind") == "chunk"]
    served = [t for t in traces if t.get("kind") != "chunk"]  # error_rate is about asks and searches
    errors = [t for t in served if t.get("status") == "error"]
    chunk_errors = [t for t in chunks if t.get("status") == "error"]
    cancelled = [t for t in traces if t.get("status") == "cancelled"]
    abstained = [t for t in asks if t.get("status") == "abstained"]
    ok_asks = [t for t in asks if t.get("status") == "ok"]
    answered = ok_asks + abstained
    ok_searches = [t for t in searches if t.get("status") == "ok"]
    usages = [t["usage"] for t in traces if isinstance(t.get("usage"), dict)]
    costs = [t.get("cost_usd") for t in traces]
    stamps = sorted(t["ts"] for t in traces if isinstance(t.get("ts"), str))
    return {
        "requests": len(traces),
        "asks": len(asks),
        "searches": len(searches),
        "chunks": len(chunks),
        "by_source": dict(Counter(t.get("source") for t in traces)),
        "by_generator": dict(Counter(t["generator"] for t in traces if t.get("generator"))),
        "abstain_rate": len(abstained) / len(asks) if asks else None,
        "error_rate": len(errors) / len(served) if served else None,
        "cancel_rate": len(cancelled) / len(traces) if traces else None,
        "latency_ms": {s: _stage(answered, s) for s in STAGES},
        "search_latency_ms": _stage(ok_searches, "total"),
        "citations_per_answer": (sum(t.get("citations") or 0 for t in ok_asks) / len(ok_asks)) if ok_asks else None,
        "tokens": {"input": _sum_known([u.get("input_tokens") for u in usages]),
                   "output": _sum_known([u.get("output_tokens") for u in usages])},
        "cost_usd": {"total": round(_sum_known(costs), 8), "unknown": sum(1 for c in costs if c is None)},
        "errors_by_type": dict(Counter(str((t.get("error") or {}).get("type", "unknown")) for t in errors)),
        "chunk_errors_by_type": dict(Counter(str((t.get("error") or {}).get("type", "unknown")) for t in chunk_errors)),
        "first_ts": stamps[0] if stamps else None,
        "last_ts": stamps[-1] if stamps else None,
    }


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _ms(x: float | None) -> str:
    return "-" if x is None else f"{x:.0f}"


def format_summary(summary: dict) -> str:
    s = summary
    errs = ", ".join(f"{k} {v}" for k, v in s["errors_by_type"].items())
    chunk_errs = ", ".join(f"{k} {v}" for k, v in s.get("chunk_errors_by_type", {}).items())
    sl = s["search_latency_ms"]
    lines = [
        f"last {s['requests']} requests ({s['first_ts']} .. {s['last_ts']}): {s['asks']} asks, {s['searches']} searches, {s['chunks']} chunks",
        f"abstain rate {_pct(s['abstain_rate'])}, error rate {_pct(s['error_rate'])}" + (f" ({errs})" if errs else "")
        + f", cancel rate {_pct(s['cancel_rate'])}",
        f"{'answer latency ms':<18}{'p50':>8}{'p95':>8}{'n':>5}",
    ]
    for stage in STAGES:
        row = s["latency_ms"][stage]
        lines.append(f"  {stage:<16}{_ms(row['p50']):>8}{_ms(row['p95']):>8}{row['n']:>5}")
    lines.append(f"{'search latency ms':<18}{_ms(sl['p50']):>8}{_ms(sl['p95']):>8}{sl['n']:>5}")
    cpa = s["citations_per_answer"]
    lines.append(
        f"citations per answer {'n/a' if cpa is None else f'{cpa:.1f}'}; tokens in {s['tokens']['input']} / "
        f"out {s['tokens']['output']}; est. cost ${s['cost_usd']['total']:.4f} ({s['cost_usd']['unknown']} unknown)"
    )
    gens = ", ".join(f"{k} {v}" for k, v in s["by_generator"].items()) or "-"
    srcs = ", ".join(f"{k} {v}" for k, v in s["by_source"].items()) or "-"
    lines.append(f"by generator: {gens} · by source: {srcs}")
    if chunk_errs:
        lines.append(f"chunk read errors: {chunk_errs}")
    return "\n".join(lines)
