"""HTTP API + static web UI.

GET  /api/health                  index and generator status
GET  /api/search?q=&k=&mode=      retrieval only (debugging, eval)
POST /api/ask {question, k}       Server-Sent Events: sources, token*, cite*, done | error
GET  /metrics?last=200            aggregates over the last N request traces (no questions, no hit details)
"""

from __future__ import annotations

import json
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

    @app.get("/metrics")
    def metrics(last: int = Query(default=200, ge=1, le=10000)) -> dict:
        if tracer is None or not tracer.enabled:
            return {"enabled": False, "last": last, "malformed_lines": 0, **summarise([])}
        traces, malformed = load_traces(tracer.directory, last)
        return {"enabled": True, "last": last, "malformed_lines": malformed, **summarise(traces)}

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
            try:
                for ev in traced_stream(pipeline, tracer, question, req.k, req.mode, source="api"):
                    yield _sse(ev)
            except Exception as e:  # never leave the client hanging on a half-open stream
                yield _sse({"type": "error", "message": f"internal error: {type(e).__name__}"})

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
    tracer = Tracer(settings.trace_path, enabled=settings.tracing_enabled, pricing=settings.pricing)
    return create_app(load_pipeline(settings), tracer)
