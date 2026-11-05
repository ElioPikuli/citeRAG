"""Shared fixtures: a tiny corpus on disk, indexed with the dependency-free HashingEmbedder."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from citerag.config import Settings
from citerag.embeddings import HashingEmbedder
from citerag.engine import ingest, load_pipeline
from citerag.generation import ExtractiveGenerator

CORPUS = {
    "proj/README.md": """\
        # Pricing Tool

        A calculator that prices concrete jobs for a construction company.

        ## Labor

        Labor days are the net volume divided by workers times cubic meters per worker per day,
        rounded up to whole days. Partial days are billed as full days.

        ## Deployment

        The app is a static site and can be deployed to GitHub Pages.
        """,
    "proj/calc.js": """\
        // Pure pricing logic.
        const EPSILON = 1e-9;

        // Partial days are billed as full days.
        function calcLaborDays(volume, workers, m3PerWorkerDay) {
          return Math.max(1, Math.ceil(volume / (workers * m3PerWorkerDay) - EPSILON));
        }

        function calcVat(amount, vatPercent) {
          return amount * vatPercent / 100;
        }
        """,
    "site/index.html": """\
        <!doctype html>
        <html><head><title>Portfolio</title>
        <style>body { color: red; }</style></head>
        <body>
        <h2>Machine Learning Models</h2>
        <p>Trained KNN and CNN models on 60K records, reaching 92% accuracy.</p>
        <script>console.log("ignored");</script>
        </body></html>
        """,
    "proj/node_modules/lib/index.js": "function shouldNotBeIndexed() {}\n",
}


def write_corpus(root: Path, files: dict[str, str] = CORPUS) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def corpus_settings(tmp_path: Path) -> Settings:
    write_corpus(tmp_path / "corpus")
    return Settings(
        root=tmp_path,
        sources=["corpus/proj", "corpus/site"],
        index_path="index",
        embedder="hashing:256",
        max_chars=400,
        overlap_chars=80,
        top_k=4,
        mode="hybrid",
        min_dense=0.9,  # hashing vectors are lexical; rely on the coverage half of the gate
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(256)


@pytest.fixture
def pipeline(corpus_settings, embedder):
    ingest(corpus_settings, embedder)
    return load_pipeline(corpus_settings, embedder=embedder, generator=ExtractiveGenerator())
