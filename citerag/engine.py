"""Composition root: settings -> index -> retriever -> pipeline."""

from __future__ import annotations

from .config import Settings
from .embeddings import Embedder, make_embedder
from .generation import Generator, default_generator
from .loaders import load_documents
from .pipeline import RagPipeline
from .retriever import Retriever
from .store import BuildReport, Index, IndexMismatch


def ingest(settings: Settings, embedder: Embedder | None = None, *, rebuild: bool = False) -> tuple[Index, BuildReport]:
    embedder = embedder or make_embedder(settings.embedder)
    previous = None
    if not rebuild and Index.exists(settings.index_dir):
        try:
            previous = Index.load(settings.index_dir, expect_embedder=embedder.name)
        except IndexMismatch:
            previous = None  # incompatible: rebuild from scratch
    docs = load_documents(settings.source_paths(), settings.include, settings.exclude)
    index, report = Index.build(
        docs, embedder, max_chars=settings.max_chars, overlap_chars=settings.overlap_chars, previous=previous
    )
    index.save(settings.index_dir)
    return index, report


def load_pipeline(settings: Settings, *, embedder: Embedder | None = None, generator: Generator | None = None,
                  index: Index | None = None) -> RagPipeline:
    embedder = embedder or make_embedder(settings.embedder)
    if index is None:
        if not Index.exists(settings.index_dir):
            raise FileNotFoundError(f"no index at {settings.index_dir}; run `citerag ingest` first")
        index = Index.load(settings.index_dir, expect_embedder=embedder.name)
    retriever = Retriever(index, embedder, candidates=settings.candidates, mmr_lambda=settings.mmr_lambda,
                          mode=settings.mode)
    return RagPipeline(
        retriever,
        generator or default_generator(settings),
        top_k=settings.top_k,
        min_coverage=settings.min_coverage,
        min_dense=settings.min_dense,
    )
