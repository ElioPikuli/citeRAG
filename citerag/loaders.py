"""Find and read source files. HTML is converted to text *line by line* so chunk line numbers still point
into the original file."""

from __future__ import annotations

import hashlib
import html
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_FILE_BYTES = 1_000_000


@dataclass(frozen=True)
class Document:
    path: Path  # absolute path on disk
    source: str  # display path, e.g. "ConcreteCalc/js/calc.js"
    text: str
    sha256: str

    @property
    def kind(self) -> str:
        return self.path.suffix.lower().lstrip(".")


def display_path(p: Path, src: Path) -> str:
    """Path shown in citations: the source folder's name plus the path inside it, e.g. 'ConcreteCalc/js/calc.js',
    independent of where the folder lives on disk."""
    return p.name if src.is_file() else f"{src.name}/{p.relative_to(src).as_posix()}"


def _included(p: Path, include: tuple[str, ...]) -> bool:
    """`include` holds extensions (".py") and exact file names for extensionless files ("Dockerfile")."""
    return p.suffix.lower() in include or p.name in include


def iter_files(sources: list[Path], include: tuple[str, ...], exclude: tuple[str, ...]):
    """Yield (file, source it came from). Missing sources are skipped with a warning, so a partial corpus
    (e.g. one repo failed to download) still indexes."""
    excluded = set(exclude)
    for src in sources:
        if src.is_file():
            if _included(src, include):
                yield src, src
            continue
        if not src.is_dir():
            print(f"warning: corpus source not found, skipping: {src}", file=sys.stderr)
            continue
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames[:] = sorted(d for d in dirnames if d not in excluded and not d.startswith("."))
            for name in sorted(filenames):
                p = Path(dirpath) / name
                if name in excluded or not _included(p, include):
                    continue
                yield p, src


def load_documents(sources: list[Path], include: tuple[str, ...], exclude: tuple[str, ...]) -> list[Document]:
    docs: list[Document] = []
    seen: set[Path] = set()
    for p, src in iter_files(sources, include, exclude):
        p, src = p.resolve(), src.resolve()
        if p in seen or p.stat().st_size > MAX_FILE_BYTES:
            continue
        seen.add(p)
        raw = p.read_bytes()
        if b"\x00" in raw[:4096]:
            continue  # binary
        text = raw.decode("utf-8", errors="replace")
        if p.suffix.lower() == ".html":
            text = html_to_text_preserving_lines(text)
        if not text.strip():
            continue
        docs.append(
            Document(path=p, source=display_path(p, src), text=text, sha256=hashlib.sha256(raw).hexdigest())
        )
    return docs


_BLOCK_STRIP = re.compile(r"<(script|style|svg|noscript)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"<h([1-6])\b[^>]*>", re.IGNORECASE)


def _blank_keep_newlines(m: re.Match) -> str:
    return "\n" * m.group(0).count("\n")


def html_to_text_preserving_lines(src: str) -> str:
    """Strip tags without changing the line count. Heading tags become Markdown '#' markers so the chunker can
    use them as section boundaries."""
    src = _COMMENT.sub(_blank_keep_newlines, src)
    src = _BLOCK_STRIP.sub(_blank_keep_newlines, src)
    # Tags can span lines (attributes on several lines): flatten each tag but keep its newlines.
    src = re.sub(r"<[^>]*\n[^>]*>", lambda m: " " * 1 + "\n" * m.group(0).count("\n"), src)
    out = []
    for line in src.split("\n"):
        h = _HEADING.search(line)
        text = html.unescape(_TAG.sub(" ", line))
        text = re.sub(r"[ \t]+", " ", text).strip()
        if h and text:
            text = "#" * int(h.group(1)) + " " + text
        out.append(text)
    return "\n".join(out)
