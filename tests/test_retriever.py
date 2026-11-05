import numpy as np
import pytest

from citerag.embeddings import HashingEmbedder
from citerag.retriever import Retriever, mmr, rrf


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[1] > fused[3] > fused[2] > fused[4]


def test_mmr_skips_near_duplicates():
    v = np.array([[1, 0], [0.999, 0.045], [0, 1]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rel = {0: 1.0, 1: 0.99, 2: 0.6}
    assert mmr([0, 1, 2], rel, v, k=2, lam=0.5) == [0, 2]
    assert mmr([0, 1, 2], rel, v, k=2, lam=1.0) == [0, 1]


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_identifier_query_finds_its_function(pipeline, mode):
    hits = pipeline.retriever.search("calcVat", k=3, mode=mode)
    assert hits[0].chunk.title.endswith("calcVat")


def test_hits_carry_per_stage_scores(pipeline):
    hit = pipeline.retriever.search("labor days rounded", k=3, mode="hybrid")[0]
    assert hit.bm25_rank is not None and hit.dense_rank is not None
    d = hit.to_dict()
    assert d["label"] == f"{hit.chunk.source}:{hit.chunk.start_line}-{hit.chunk.end_line}"


def test_excluded_and_html_noise_not_indexed(pipeline):
    texts = " ".join(c.text for c in pipeline.retriever.index.chunks)
    assert "shouldNotBeIndexed" not in texts
    assert "console.log" not in texts and "color: red" not in texts


def test_empty_query_returns_nothing(pipeline):
    assert pipeline.retriever.search("   ") == []


def test_embedder_mismatch_is_rejected(pipeline):
    with pytest.raises(ValueError):
        Retriever(pipeline.retriever.index, HashingEmbedder(128))
