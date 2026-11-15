"""Request tracing: one JSON line per ask/search in {trace_dir}/{YYYY-MM-DD}.jsonl.

Tracing must never break a request, so every write failure is logged and swallowed. Traces hold the question
truncated to 300 characters and hit locations, never chunk text; `summarise` goes further and keeps only aggregates,
because /metrics is served publicly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .retriever import Hit
from .stats import percentile

log = logging.getLogger("citerag.tracing")

QUESTION_MAX = 300
HITS_MAX = 10
FREE_GENERATORS = ("extractive", "abstain", "ollama")
STAGES = ("retrieve", "gate", "generate", "total")


class Tracer:
    def __init__(self, directory: Path, *, enabled: bool = True, pricing: dict[str, dict] | None = None):
        self.directory = Path(directory)
        self.enabled = enabled
        self.pricing = pricing or {}
        self._lock = threading.Lock()  # FastAPI runs sync endpoints in a threadpool

    def write(self, trace: dict) -> None:
        """Append one line to today's file (UTC). Never raises."""
        try:
            line = json.dumps(trace, ensure_ascii=False)
            path = self.directory / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as e:  # noqa: BLE001 - tracing must never break a request
            log.warning("could not write trace: %s: %s", type(e).__name__, e)


def estimate_cost(generator: str | None, model: str | None, usage: dict | None, pricing: dict) -> float | None:
    """USD estimate. Free generators cost 0.0; an unpriced model or unreported tokens give None, never 0."""
    if generator in FREE_GENERATORS:
        return 0.0
    price = pricing.get(model) if model else None
    if not price or not usage:
        return None
    tokens_in, tokens_out = usage.get("input_tokens"), usage.get("output_tokens")
    if tokens_in is None or tokens_out is None or "input" not in price or "output" not in price:
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


def traced_stream(pipeline, tracer: Tracer | None, question: str, k: int | None = None, mode: str | None = None,
                  *, source: str = "api") -> Iterator[dict]:
    """Pass every event of pipeline.stream through unchanged and write one trace when the stream ends,
    however it ends (done, error event, exception, or the consumer closing the generator)."""
    if tracer is None or not tracer.enabled:
        yield from pipeline.stream(question, k, mode)
        return
    trace = _blank("ask", source, question, k, mode)
    cited: set[int] = set()
    done: dict | None = None
    t0 = time.perf_counter()
    inner = pipeline.stream(question, k, mode)
    try:
        for ev in inner:
            kind = ev.get("type")
            if kind == "sources":
                trace["hits"] = _rows_from_dicts(ev.get("hits", []))
                gate = ev.get("gate") or {}
                trace["gate"] = {"answerable": gate.get("answerable"), "coverage": gate.get("coverage"),
                                 "max_dense": gate.get("max_dense")}
                trace["mode"] = gate.get("mode", mode)
                trace["latency_ms"]["retrieve"] = gate.get("retrieval_ms")
                trace["latency_ms"]["gate"] = gate.get("gate_ms")
            elif kind == "cite":
                trace["cite_events"] += 1
                cited.update(ev.get("hits", []))
            elif kind == "done":
                done = ev
            elif kind == "error":
                trace["error"] = {"type": "generator_error", "message": str(ev.get("message"))[:300]}
            yield ev
    except GeneratorExit:
        if done is None and trace["error"] is None:
            trace["error"] = {"type": "client_disconnected", "message": "client closed the stream before done"}
        raise
    except Exception as e:
        trace["error"] = {"type": type(e).__name__, "message": str(e)[:300]}
        raise
    finally:
        try:
            inner.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            _finish_ask(trace, done, cited, pipeline, tracer, (time.perf_counter() - t0) * 1000)
            tracer.write(trace)
        except Exception as e:  # noqa: BLE001 - never break the request
            log.warning("could not build trace: %s: %s", type(e).__name__, e)


def _finish_ask(trace: dict, done: dict | None, cited: set[int], pipeline, tracer: Tracer, total_ms: float) -> None:
    trace["citations"] = len(cited)
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
    trace["cost_usd"] = estimate_cost(trace["generator"], trace["model"], trace["usage"], tracer.pricing)


def traced_search(pipeline, tracer: Tracer | None, query: str, k: int, mode: str | None = None,
                  *, source: str = "api") -> list[Hit]:
    """pipeline.retriever.search with a trace. Exceptions are recorded and re-raised."""
    if tracer is None or not tracer.enabled:
        return pipeline.retriever.search(query, k, mode=mode)
    trace = _blank("search", source, query, k, mode or pipeline.retriever.mode)
    trace["cost_usd"] = 0.0  # retrieval only: local embeddings and BM25, no model call
    t0 = time.perf_counter()
    try:
        hits = pipeline.retriever.search(query, k, mode=mode)
        trace["hits"] = _rows_from_hits(hits)
        return hits
    except Exception as e:
        trace["error"] = {"type": type(e).__name__, "message": str(e)[:300]}
        trace["status"] = "error"
        raise
    finally:
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        trace["latency_ms"]["retrieve"] = elapsed
        trace["latency_ms"]["total"] = elapsed
        tracer.write(trace)


def load_traces(directory: Path, last: int = 200) -> tuple[list[dict], int]:
    """Newest `last` traces (returned oldest first) and the number of malformed lines skipped on the way."""
    directory = Path(directory)
    if not directory.is_dir():
        return [], 0
    out: list[dict] = []
    malformed = 0
    for path in sorted(directory.glob("????-??-??.jsonl"), reverse=True):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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
            if isinstance(obj, dict):
                out.append(obj)
            else:
                malformed += 1
        if len(out) >= last:
            break
    out.reverse()
    return out, malformed


def _stage(traces: list[dict], stage: str) -> dict:
    vals = [t["latency_ms"][stage] for t in traces
            if isinstance(t.get("latency_ms"), dict) and isinstance(t["latency_ms"].get(stage), (int, float))]
    return {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "n": len(vals)}


def _sum_known(values: list) -> int:
    return sum(v for v in values if isinstance(v, (int, float)))


def summarise(traces: list[dict]) -> dict:
    """Aggregates only: no question text, no hit details."""
    asks = [t for t in traces if t.get("kind") == "ask"]
    searches = [t for t in traces if t.get("kind") == "search"]
    errors = [t for t in traces if t.get("status") == "error"]
    abstained = [t for t in asks if t.get("status") == "abstained"]
    ok_asks = [t for t in asks if t.get("status") == "ok"]
    usages = [t["usage"] for t in traces if isinstance(t.get("usage"), dict)]
    costs = [t.get("cost_usd") for t in traces]
    stamps = sorted(t["ts"] for t in traces if isinstance(t.get("ts"), str))
    return {
        "requests": len(traces),
        "asks": len(asks),
        "searches": len(searches),
        "by_source": dict(Counter(t.get("source") for t in traces)),
        "by_generator": dict(Counter(t["generator"] for t in traces if t.get("generator"))),
        "abstain_rate": len(abstained) / len(asks) if asks else None,
        "error_rate": len(errors) / len(traces) if traces else None,
        "latency_ms": {s: _stage(traces, s) for s in STAGES},
        "citations_per_answer": (sum(t.get("citations") or 0 for t in ok_asks) / len(ok_asks)) if ok_asks else None,
        "tokens": {"input": _sum_known([u.get("input_tokens") for u in usages]),
                   "output": _sum_known([u.get("output_tokens") for u in usages])},
        "cost_usd": {"total": round(_sum_known(costs), 8), "unknown": sum(1 for c in costs if c is None)},
        "errors_by_type": dict(Counter((t.get("error") or {}).get("type", "unknown") for t in errors)),
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
    lines = [
        f"last {s['requests']} requests ({s['first_ts']} .. {s['last_ts']}): {s['asks']} asks, {s['searches']} searches",
        f"abstain rate {_pct(s['abstain_rate'])}, error rate {_pct(s['error_rate'])}" + (f" ({errs})" if errs else ""),
        f"{'latency ms':<14}{'p50':>8}{'p95':>8}{'n':>5}",
    ]
    for stage in STAGES:
        row = s["latency_ms"][stage]
        lines.append(f"  {stage:<12}{_ms(row['p50']):>8}{_ms(row['p95']):>8}{row['n']:>5}")
    cpa = s["citations_per_answer"]
    lines.append(
        f"citations per answer {'n/a' if cpa is None else f'{cpa:.1f}'}; tokens in {s['tokens']['input']} / "
        f"out {s['tokens']['output']}; est. cost ${s['cost_usd']['total']:.4f} ({s['cost_usd']['unknown']} unknown)"
    )
    gens = ", ".join(f"{k} {v}" for k, v in s["by_generator"].items()) or "-"
    srcs = ", ".join(f"{k} {v}" for k, v in s["by_source"].items()) or "-"
    lines.append(f"by generator: {gens} · by source: {srcs}")
    return "\n".join(lines)
