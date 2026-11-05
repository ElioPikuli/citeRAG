"""The persisted index: chunks + dense vectors + BM25, with incremental rebuilds.

Layout of the index directory:
    manifest.json   embedder name, chunking params, sha256 of every indexed file
    chunks.jsonl    one Chunk per line, row-aligned with vectors.npy
    vectors.npy     float32 [n_chunks, dim], L2-normalized
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunking import CHUNKER_VERSION, Chunk, chunk_document
from .embeddings import Embedder
from .loaders import Document
from .text import tokenize

FORMAT_VERSION = 1


class IndexMismatch(RuntimeError):
    pass


@dataclass
class BuildReport:
    files_total: int = 0
    files_changed: int = 0
    files_removed: int = 0
    chunks_total: int = 0
    chunks_embedded: int = 0
    seconds: float = 0.0


@dataclass
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    embedder_name: str
    chunking: dict
    files: dict[str, str] = field(default_factory=dict)  # source -> sha256
    _bm25: BM25 | None = field(default=None, repr=False)

    def __post_init__(self):
        if len(self.chunks) != len(self.vectors):
            raise ValueError("chunks and vectors are not aligned")

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25([tokenize(c.indexed_text()) for c in self.chunks])
        return self._bm25

    def __len__(self) -> int:
        return len(self.chunks)

    # ---------- build ----------

    @classmethod
    def build(
        cls,
        docs: list[Document],
        embedder: Embedder,
        *,
        max_chars: int = 1200,
        overlap_chars: int = 200,
        previous: "Index | None" = None,
    ) -> tuple["Index", BuildReport]:
        """Chunk and embed `docs`. Vectors of files whose sha256 is unchanged since `previous` are reused."""
        t0 = time.perf_counter()
        chunking = {"max_chars": max_chars, "overlap_chars": overlap_chars, "version": CHUNKER_VERSION}
        reusable = (
            previous is not None and previous.embedder_name == embedder.name and previous.chunking == chunking
        )
        prev_by_source: dict[str, list[tuple[Chunk, np.ndarray]]] = {}
        if reusable:
            for c, v in zip(previous.chunks, previous.vectors):
                prev_by_source.setdefault(c.source, []).append((c, v))

        report = BuildReport(files_total=len(docs))
        chunks: list[Chunk] = []
        vectors: list[np.ndarray | None] = []
        to_embed: list[int] = []
        for doc in docs:
            if reusable and previous.files.get(doc.source) == doc.sha256:
                for c, v in prev_by_source.get(doc.source, []):
                    chunks.append(c)
                    vectors.append(v)
                continue
            report.files_changed += 1
            for c in chunk_document(doc, max_chars, overlap_chars):
                to_embed.append(len(chunks))
                chunks.append(c)
                vectors.append(None)

        if to_embed:
            fresh = embedder.embed_documents([chunks[i].indexed_text() for i in to_embed])
            for row, i in enumerate(to_embed):
                vectors[i] = fresh[row]
        matrix = np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, embedder.dim), np.float32)

        if previous is not None:
            current = {d.source for d in docs}
            report.files_removed = sum(1 for s in previous.files if s not in current)
        report.chunks_total = len(chunks)
        report.chunks_embedded = len(to_embed)
        report.seconds = round(time.perf_counter() - t0, 2)
        files = {d.source: d.sha256 for d in docs}
        return cls(chunks, matrix, embedder.name, chunking, files), report

    # ---------- persistence ----------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = directory / ".tmp"
        tmp.mkdir(exist_ok=True)
        with open(tmp / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        np.save(tmp / "vectors.npy", self.vectors)
        manifest = {
            "format": FORMAT_VERSION,
            "embedder": self.embedder_name,
            "dim": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0,
            "chunking": self.chunking,
            "files": self.files,
            "n_chunks": len(self.chunks),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        # Replace files only after everything is written, so a crash never leaves a half-written index.
        for name in ("chunks.jsonl", "vectors.npy", "manifest.json"):
            (tmp / name).replace(directory / name)
        tmp.rmdir()

    @classmethod
    def exists(cls, directory: Path) -> bool:
        return (directory / "manifest.json").exists()

    @classmethod
    def load(cls, directory: Path, expect_embedder: str | None = None) -> "Index":
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT_VERSION:
            raise IndexMismatch("index format changed; run `citerag ingest --rebuild`")
        if expect_embedder and manifest["embedder"] != expect_embedder:
            raise IndexMismatch(
                f"index was built with {manifest['embedder']!r} but config uses {expect_embedder!r}; "
                "run `citerag ingest --rebuild`"
            )
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        vectors = np.load(directory / "vectors.npy")
        return cls(chunks, vectors, manifest["embedder"], manifest["chunking"], manifest["files"])
