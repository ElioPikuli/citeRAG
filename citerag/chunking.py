"""Structure-aware chunking.

1. Split a document into *sections*: Markdown/HTML headings, or top-level definitions in code.
2. Split each section into *units*: paragraphs; fenced code blocks stay whole.
3. Pack units greedily up to `max_chars`, carrying `overlap_chars` of trailing lines into the next chunk.

Chunks never cross a section boundary, and each one records its 1-based inclusive line range.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .loaders import Document

# Bump when chunking logic changes: the store re-chunks every file when this differs from the index.
CHUNKER_VERSION = 4

CODE_KINDS = {"py", "js", "ts", "jsx", "tsx", "cs"}
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
# Section starts: functions/classes up to one indent level deep (code is often wrapped in an IIFE or a class),
# and top-level const/let/var bindings.
_CODE_DEF = re.compile(
    r"^(?:\s{0,4}(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:def|class|function\*?)"
    r"|(?:export\s+)?(?:const|let|var))\s+([A-Za-z_$][\w$]*)"
)
# C#: type declarations, and method/constructor declarations (a signature line with "(" and no ";", so calls and
# field initializers don't match). Up to two indent levels: namespace > class > member.
_CS_MODS = r"(?:(?:public|private|protected|internal|static|override|virtual|abstract|async|sealed|partial|new|unsafe|extern)\s+)*"
_CS_TYPE = re.compile(rf"^\s{{0,4}}{_CS_MODS}(?:class|struct|interface|enum|record)\s+(\w+)")
_CS_METHOD = re.compile(rf"^\s{{0,8}}{_CS_MODS}(?:[\w<>\[\],.?]+\s+)?(\w+)\s*\([^;]*$")
_CS_NOT_METHOD = {"if", "for", "foreach", "while", "switch", "catch", "using", "lock", "return", "else", "new",
                  "nameof", "typeof", "base", "this"}


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str
    start_line: int
    end_line: int

    @property
    def label(self) -> str:
        return f"{self.source}:{self.start_line}-{self.end_line}"

    def indexed_text(self) -> str:
        """What retrieval sees: the section path gives short chunks their context."""
        return f"{self.source} > {self.title}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Section:
    title: str
    lines: list[tuple[int, str]]  # (line_no, text)


def _sections(doc: Document) -> list[_Section]:
    lines = doc.text.split("\n")
    file_title = doc.source.rsplit("/", 1)[-1]
    if doc.kind in CODE_KINDS:
        return _code_sections(lines, file_title, doc.kind)
    return _heading_sections(lines, file_title)


def _heading_sections(lines: list[str], file_title: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(file_title, [])
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _MD_HEADING.match(line)
        if m:
            if current.lines:
                sections.append(current)
            level, name = len(m.group(1)), m.group(2).strip()
            stack = [(lv, n) for lv, n in stack if lv < level] + [(level, name)]
            current = _Section(" > ".join(n for _, n in stack), [])
        current.lines.append((i, line))
    if current.lines:
        sections.append(current)
    return sections


def _cs_def(line: str, cls: str | None) -> tuple[str, str | None] | None:
    """(section name, enclosing class) for a C# declaration line, else None."""
    if m := _CS_TYPE.match(line):
        return m.group(1), m.group(1)
    m = _CS_METHOD.match(line)
    if m and m.group(1) not in _CS_NOT_METHOD and not line.lstrip().startswith(tuple(_CS_NOT_METHOD)):
        return (f"{cls}.{m.group(1)}" if cls else m.group(1)), cls
    return None


def _code_sections(lines: list[str], file_title: str, kind: str = "") -> list[_Section]:
    sections: list[_Section] = []
    current = _Section(file_title, [])
    pending: list[tuple[int, str]] = []  # comments / decorators / attributes directly above a definition
    attach = ("#", "//", "/*", "*", "@", "[") if kind == "cs" else ("#", "//", "/*", "*", "@")
    cls: str | None = None
    for i, line in enumerate(lines, start=1):
        name = None
        if kind == "cs":
            found = _cs_def(line, cls)
            if found:
                name, cls = found
        elif m := _CODE_DEF.match(line):
            name = m.group(1)
        if name:
            if current.lines:
                sections.append(current)
            current = _Section(f"{file_title} > {name}", pending)
            pending = []
            current.lines.append((i, line))
            continue
        stripped = line.strip()
        is_attachable = stripped.startswith(attach) and len(line) - len(line.lstrip()) <= 4
        if is_attachable:
            pending.append((i, line))
            continue
        if pending:
            current.lines.extend(pending)
            pending = []
        current.lines.append((i, line))
    current.lines.extend(pending)
    if current.lines:
        sections.append(current)
    return sections


def _units(section: _Section) -> list[list[tuple[int, str]]]:
    """Paragraphs separated by blank lines; a fenced code block is kept as one unit."""
    units: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    in_fence = False
    for no, line in section.lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if cur:
                units.append(cur)
                cur = []
            continue
        cur.append((no, line))
    if cur:
        units.append(cur)
    return units


def _split_long(unit: list[tuple[int, str]], max_chars: int) -> list[list[tuple[int, str]]]:
    parts: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    size = 0
    for no, line in unit:
        for start in range(0, max(len(line), 1), max_chars):
            piece = line[start : start + max_chars]
            if cur and size + len(piece) + 1 > max_chars:
                parts.append(cur)
                cur, size = [], 0
            cur.append((no, piece))
            size += len(piece) + 1
    if cur:
        parts.append(cur)
    return parts


def _size(lines: list[tuple[int, str]]) -> int:
    return sum(len(t) + 1 for _, t in lines)


def chunk_document(doc: Document, max_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    chunks: list[Chunk] = []
    for section in _sections(doc):
        units: list[list[tuple[int, str]]] = []
        for u in _units(section):
            units.extend(_split_long(u, max_chars) if _size(u) > max_chars else [u])
        cur: list[tuple[int, str]] = []
        fresh = False  # does `cur` hold anything beyond overlap carried from the previous chunk?
        for u in units:
            if cur and fresh and _size(cur) + _size(u) > max_chars:
                chunks.append(_make_chunk(doc, section.title, cur))
                cur = _tail(cur, overlap_chars)
                fresh = False
            # overlap must never push a chunk over the limit
            while cur and _size(cur) + _size(u) > max_chars:
                cur = cur[1:]
            cur = cur + u
            fresh = True
        if cur and fresh:
            chunks.append(_make_chunk(doc, section.title, cur))
    # A heading with no body carries no information of its own (its text lives on in child titles).
    return [c for c in chunks if not (c.start_line == c.end_line and _MD_HEADING.match(c.text))]


def _tail(lines: list[tuple[int, str]], budget: int) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    size = 0
    for no, text in reversed(lines):
        if size + len(text) + 1 > budget:
            break
        out.insert(0, (no, text))
        size += len(text) + 1
    return out


def _make_chunk(doc: Document, title: str, lines: list[tuple[int, str]]) -> Chunk:
    text = "\n".join(t for _, t in lines).strip("\n").rstrip()  # keep the first line's indentation
    start, end = lines[0][0], lines[-1][0]
    cid = hashlib.sha1(f"{doc.source}:{start}:{end}:{text}".encode()).hexdigest()[:16]
    return Chunk(id=cid, source=doc.source, title=title, text=text, start_line=start, end_line=end)
