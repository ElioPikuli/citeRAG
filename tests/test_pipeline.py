import pytest

from citerag.pipeline import ABSTAIN_MESSAGE


def test_answers_on_topic_question_with_citations(pipeline):
    ans = pipeline.answer("How are partial labor days billed?")
    assert not ans.abstained and ans.generator == "extractive"
    assert ans.citations
    assert any(h.chunk.source.endswith(("README.md", "calc.js")) for h in ans.cited_hits())
    assert ans.meta["gate"]["answerable"] is True


def test_abstains_off_topic_without_calling_generator(pipeline):
    class Boom:
        name = "boom"

        def stream(self, *a):
            raise AssertionError("generator must not be called")

    pipeline.generator = Boom()
    ans = pipeline.answer("What is the capital of Australia?")
    assert ans.abstained and ans.text == ABSTAIN_MESSAGE
    assert ans.meta["gate"]["coverage"] < pipeline.min_coverage


def test_stream_event_order(pipeline):
    types = [e["type"] for e in pipeline.stream("What accuracy did the models reach?")]
    assert types[0] == "sources" and types[-1] == "done"
    assert "token" in types


def test_term_coverage_weights_rare_terms(pipeline):
    hits = pipeline.retriever.search("calcVat", k=4)
    assert pipeline.term_coverage("calcVat", hits) == 1.0
    assert pipeline.term_coverage("calcVat kangaroo", hits) == pytest.approx(0.5, abs=0.15)
    assert pipeline.term_coverage("kangaroo wombat", hits) == 0.0
