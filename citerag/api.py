"""HTTP API + static web UI.

GET  /api/health                  index and generator status
GET  /api/search?q=&k=&mode=      retrieval only (debugging, eval)
POST /api/ask {question, k}       Server-Sent Events: sources, token*, cite*, done | error
GET  /metrics?last=200            aggregates over the last N request traces (no questions, no hit details)
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import RagPipeline
from .tracing import Tracer, load_traces, summarise, traced_search, traced_stream

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

EXAMPLE_QUESTIONS = [
    "How does the concrete calculator work out the number of labor days?",
    "Which projects use machine learning, and what accuracy did they reach?",
    "How is the Concrete Quote Calculator tested?",
    "What is Dynamic Labyrinths about?",
    "Why does this RAG system fuse BM25 and embeddings by rank?",
    "What does safeFileName do?",
]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=20)
    mode: Literal["hybrid", "bm25", "dense"] | None = None


METRICS_TTL_S = 5.0
METRICS_MAX_LAST = 1000


def _now() -> float:
    return time.monotonic()


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def create_app(pipeline: RagPipeline, tracer: Tracer | None = None) -> FastAPI:
    """`tracer=None` turns tracing off, so tests never write into the repo."""
    app = FastAPI(title="CiteRAG", version="0.1.0")
    index = pipeline.retriever.index

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "chunks": len(index),
            "files": len(index.files),
            "embedder": index.embedder_name,
            "generator": pipeline.generator.name,
            "mode": pipeline.retriever.mode,
            "tracing": bool(tracer and tracer.enabled),
        }

    metrics_cache: dict[int, tuple[float, dict]] = {}
    metrics_lock = threading.Lock()

    @app.get("/metrics")
    def metrics(last: int = Query(default=200, ge=1, le=METRICS_MAX_LAST)) -> dict:
        if tracer is None or not tracer.enabled:
            return {"enabled": False, "last": last, "malformed_lines": 0, **summarise([])}
        # Public endpoint that reads files: recompute at most once per TTL per `last` value.
        with metrics_lock:
            cached = metrics_cache.get(last)
            if cached and _now() - cached[0] < METRICS_TTL_S:
                return cached[1]
            traces, malformed = load_traces(tracer.directory, last)
            payload = {"enabled": True, "last": last, "malformed_lines": malformed, **summarise(traces)}
            metrics_cache[last] = (_now(), payload)
            return payload

    @app.get("/api/examples")
    def examples() -> list[str]:
        return EXAMPLE_QUESTIONS

    @app.get("/api/search")
    def search(
        q: str = Query(min_length=1, max_length=1000),
        k: int = Query(default=6, ge=1, le=50),
        mode: Literal["hybrid", "bm25", "dense"] | None = None,
    ) -> dict:
        hits = traced_search(pipeline, tracer, q, k, mode, source="api")
        return {"query": q, "mode": mode or pipeline.retriever.mode, "hits": [h.to_dict() for h in hits]}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is empty")

        def events() -> Iterator[str]:
            stream = traced_stream(pipeline, tracer, question, req.k, req.mode, source="api")
            try:
                for ev in stream:
                    yield _sse(ev)
            except Exception as e:  # never leave the client hanging on a half-open stream
                yield _sse({"type": "error", "message": f"internal error: {type(e).__name__}"})
            finally:
                stream.close()  # a disconnect closes this generator; make the trace see it now, not at GC

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def home() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def app_from_settings(config: str = "citerag.toml") -> FastAPI:
    from .config import Settings
    from .engine import load_pipeline

    settings = Settings.load(config)
    tracer = Tracer(settings.trace_path, enabled=settings.tracing_enabled, pricing=settings.pricing,
                  max_file_mb=settings.trace_max_file_mb, retention_days=settings.trace_retention_days)
    return create_app(load_pipeline(settings), tracer)
