"""Tokenization shared by BM25, the abstain gate and the extractive generator.

Code-aware: `calcLaborDays` -> calc, labor, days (plus the whole identifier), `include_waste` -> include, waste.
Unicode-aware, so Hebrew and other scripts tokenize too.
"""

from __future__ import annotations

import re

WORD = _WORD = re.compile(r"\w+", re.UNICODE)
# Hebrew abbreviations use gershayim/geresh inside the word (מ״ק, ע"מ): join them so the word stays one token.
_HE_ABBREV = re.compile(r"(?<=[\u05D0-\u05EA])[\u05F4\u05F3\"'](?=[\u05D0-\u05EA])")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from has have how i if in into is it its of on or
    that the their them then there these this to was what when where which who why will with you your
    """.split()
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for raw in _WORD.findall(_HE_ABBREV.sub("", text)):
        parts = [p for piece in raw.split("_") for p in _CAMEL.findall(piece)] if raw.isascii() else []
        lowered = raw.lower()
        if len(parts) > 1:
            out.append(lowered)  # keep the whole identifier for exact matches
            out.extend(p.lower() for p in parts)
        else:
            out.append(lowered)
    if keep_stopwords:
        return [stem(t) for t in out]
    return [stem(t) for t in out if (t not in STOPWORDS and len(t) > 1) or t.isdigit()]


def stem(t: str) -> str:
    """Plural folding only ("defaults" -> "default", "queries" -> "query"). Deliberately light: aggressive
    stemmers mangle code identifiers, which are exactly what BM25 is here to match."""
    if len(t) <= 3 or not t.isascii() or not t.isalpha():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("sses", "xes", "ches", "shes")):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[\u05D0-\u05EA])|\n{2,}|\n(?=\s*[-*|#] )")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s and s.strip()]
