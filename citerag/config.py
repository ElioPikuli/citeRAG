"""Settings, loaded from citerag.toml. Every field has a default so tests can build Settings() directly."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_INCLUDE = (".md", ".txt", ".html", ".py", ".js", ".ts", ".css", ".json", ".toml")
DEFAULT_EXCLUDE = ("node_modules", ".git", ".venv", "__pycache__", "package-lock.json")


@dataclass
class Settings:
    root: Path = field(default_factory=Path.cwd)
    sources: list[str] = field(default_factory=lambda: ["."])
    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE
    index_path: str = "index"
    embedder: str = "fastembed:BAAI/bge-small-en-v1.5"
    max_chars: int = 1200
    overlap_chars: int = 200
    mode: str = "hybrid"
    top_k: int = 6
    candidates: int = 30
    mmr_lambda: float = 0.7
    min_coverage: float = 0.8
    min_dense: float = 0.555
    model: str = "claude-opus-5-5"
    effort: str = "low"
    max_tokens: int = 4096

    @property
    def index_dir(self) -> Path:
        return (self.root / self.index_path).resolve()

    def source_paths(self) -> list[Path]:
        return [(self.root / s).resolve() for s in self.sources]

    @classmethod
    def load(cls, path: str | Path = "citerag.toml") -> "Settings":
        path = Path(path).resolve()
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        corpus, index = data.get("corpus", {}), data.get("index", {})
        chunking, retrieval = data.get("chunking", {}), data.get("retrieval", {})
        gen = data.get("generation", {})
        d = cls()
        return cls(
            root=path.parent,
            sources=corpus.get("sources", d.sources),
            include=tuple(corpus.get("include", d.include)),
            exclude=tuple(corpus.get("exclude", d.exclude)),
            index_path=index.get("path", d.index_path),
            embedder=index.get("embedder", d.embedder),
            max_chars=chunking.get("max_chars", d.max_chars),
            overlap_chars=chunking.get("overlap_chars", d.overlap_chars),
            mode=retrieval.get("mode", d.mode),
            top_k=retrieval.get("top_k", d.top_k),
            candidates=retrieval.get("candidates", d.candidates),
            mmr_lambda=retrieval.get("mmr_lambda", d.mmr_lambda),
            min_coverage=retrieval.get("min_coverage", d.min_coverage),
            min_dense=retrieval.get("min_dense", d.min_dense),
            model=gen.get("model", d.model),
            effort=gen.get("effort", d.effort),
            max_tokens=gen.get("max_tokens", d.max_tokens),
        )
