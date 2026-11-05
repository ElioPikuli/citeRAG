import math

import pytest

from citerag.bm25 import BM25


def test_matches_hand_computed_score():
    docs = [["a", "b"], ["a", "c", "c"], ["d"]]
    bm = BM25(docs, k1=1.5, b=0.75)
    avgdl = 2.0
    idf_c = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl)
    expected = idf_c * 2 * 2.5 / (2 + norm)
    assert bm.scores(["c"])[1] == pytest.approx(expected)
    assert bm.scores(["c"])[0] == 0.0


def test_rarer_terms_score_higher_and_unknown_terms_are_ignored():
    bm = BM25([["common", "rare"], ["common"], ["common"]])
    assert bm.idf["rare"] > bm.idf["common"] > 0
    assert bm.top(["rare", "zzz"], 5) == [(0, pytest.approx(bm.scores(["rare"])[0]))]


def test_empty_index():
    assert BM25([]).top(["x"], 3) == []
