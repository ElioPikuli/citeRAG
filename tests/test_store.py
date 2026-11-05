import pytest

from citerag.embeddings import HashingEmbedder
from citerag.engine import ingest
from citerag.store import Index, IndexMismatch


class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__(256)
        self.embedded = 0

    def embed_documents(self, texts):
        self.embedded += len(texts)
        return super().embed_documents(texts)


def test_round_trip(corpus_settings, embedder):
    index, report = ingest(corpus_settings, embedder)
    loaded = Index.load(corpus_settings.index_dir, expect_embedder=embedder.name)
    assert [c.to_dict() for c in loaded.chunks] == [c.to_dict() for c in index.chunks]
    assert (loaded.vectors == index.vectors).all()
    assert report.files_total == 3 and report.chunks_total == len(index) > 0


def test_incremental_reembeds_only_changed_files(corpus_settings):
    e = CountingEmbedder()
    ingest(corpus_settings, e)
    first = e.embedded

    e.embedded = 0
    _, report = ingest(corpus_settings, e)
    assert e.embedded == 0 and report.files_changed == 0

    readme = corpus_settings.root / "corpus/proj/README.md"
    readme.write_text(readme.read_text() + "\n## New\nA new section about invoices.\n")
    index, report = ingest(corpus_settings, e)
    assert report.files_changed == 1
    assert 0 < e.embedded < first
    assert any("invoices" in c.text for c in index.chunks)

    (corpus_settings.root / "corpus/site/index.html").unlink()
    index, report = ingest(corpus_settings, e)
    assert report.files_removed == 1
    assert not any(c.source.endswith("index.html") for c in index.chunks)


def test_chunking_change_forces_full_rebuild(corpus_settings):
    e = CountingEmbedder()
    ingest(corpus_settings, e)
    e.embedded = 0
    corpus_settings.max_chars = 300
    index, _ = ingest(corpus_settings, e)
    assert e.embedded == len(index)


def test_loading_with_other_embedder_fails(corpus_settings, embedder):
    ingest(corpus_settings, embedder)
    with pytest.raises(IndexMismatch):
        Index.load(corpus_settings.index_dir, expect_embedder="fastembed:other")
