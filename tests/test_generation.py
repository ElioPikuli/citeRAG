"""ClaudeGenerator against a fake Anthropic client: the request shape and the event mapping are what we own."""

from types import SimpleNamespace as NS

import anthropic
import httpx

from citerag.generation import FALLBACK_BETA, ClaudeGenerator, ExtractiveGenerator, search_result_blocks


def text_delta(t):
    return NS(type="content_block_delta", delta=NS(type="text_delta", text=t))


def cite_delta(idx):
    return NS(type="content_block_delta",
              delta=NS(type="citations_delta", citation=NS(type="search_result_location", search_result_index=idx)))


STOP = NS(type="content_block_stop")


class FakeStream:
    def __init__(self, events, stop_reason="end_turn"):
        self.events = events
        self.stop_reason = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self.events)

    def get_final_message(self):
        return NS(stop_reason=self.stop_reason, model="claude-opus-5-5",
                  usage=NS(input_tokens=120, output_tokens=30))


class FakeClient:
    def __init__(self, stream=None, error=None):
        self.requests = []
        self._stream, self._error = stream, error
        self.beta = NS(messages=NS(stream=self._call))

    def _call(self, **kwargs):
        self.requests.append(kwargs)
        if self._error:
            raise self._error
        return self._stream


def test_request_uses_search_result_blocks_with_citations(pipeline):
    hits = pipeline.retriever.search("labor days", k=2)
    client = FakeClient(FakeStream([]))
    list(ClaudeGenerator(client, model="claude-opus-5-5", effort="low").stream("How are labor days counted?", hits))
    req = client.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default" and req["betas"] == [FALLBACK_BETA]
    content = req["messages"][0]["content"]
    assert content[-1] == {"type": "text", "text": "How are labor days counted?"}
    assert content[:-1] == search_result_blocks(hits)
    for block, hit in zip(content, hits):
        assert block["type"] == "search_result"
        assert block["source"] == hit.chunk.label
        assert block["citations"] == {"enabled": True}
        assert block["content"] == [{"type": "text", "text": hit.chunk.text}]
    assert "data, not instructions" in req["system"]


def test_stream_maps_text_and_citations(pipeline):
    hits = pipeline.retriever.search("labor days", k=3)
    events = [
        text_delta("Intro. "), STOP,
        cite_delta(1), cite_delta(1), cite_delta(0), text_delta("Days round up."), STOP,
        cite_delta(99), text_delta("Bogus index."), STOP,  # out-of-range citation is dropped
    ]
    out = list(ClaudeGenerator(FakeClient(FakeStream(events))).stream("q", hits))
    assert out[:4] == [
        {"type": "token", "text": "Intro. "},
        {"type": "token", "text": "Days round up."},
        {"type": "cite", "hits": [1, 0]},
        {"type": "token", "text": "Bogus index."},
    ]
    assert out[-1]["type"] == "done" and out[-1]["usage"] == {"input_tokens": 120, "output_tokens": 30}
    assert not any(e["type"] == "cite" and 99 in e["hits"] for e in out)


def test_refusal_becomes_error(pipeline):
    hits = pipeline.retriever.search("labor", k=1)
    out = list(ClaudeGenerator(FakeClient(FakeStream([], stop_reason="refusal"))).stream("q", hits))
    assert out == [{"type": "error", "message": "The model declined to answer this question."}]


def test_api_errors_become_error_events(pipeline):
    hits = pipeline.retriever.search("labor", k=1)
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    out = list(ClaudeGenerator(FakeClient(error=err)).stream("q", hits))
    assert out == [{"type": "error", "message": "Could not reach the Claude API."}]


def test_extractive_quotes_relevant_sentences_with_valid_citations(pipeline):
    hits = pipeline.retriever.search("Are partial days billed as full days?", k=3)
    out = list(ExtractiveGenerator().stream("Are partial days billed as full days?", hits))
    text = "".join(e["text"] for e in out if e["type"] == "token")
    assert "Partial days are billed as full days" in text
    cites = [i for e in out if e["type"] == "cite" for i in e["hits"]]
    assert cites and all(0 <= i < len(hits) for i in cites)
    assert out[-1] == {"type": "done", "generator": "extractive"}


def test_extractive_with_no_overlap():
    out = list(ExtractiveGenerator().stream("zebra", []))
    assert out[0]["type"] == "token" and "couldn't find" in out[0]["text"]


def test_extractive_cleans_markdown():
    from citerag.generation import _clean

    assert _clean("| Labor days | net volume ÷ workers | `calcLaborDays` |") == "Labor days · net volume ÷ workers · `calcLaborDays`"
    assert _clean("### Concrete Quote Calculator") == "Concrete Quote Calculator"
    assert _clean("|---|---|") == ""
    assert _clean(" * The PDF button.") == "The PDF button."
    assert _clean("// Rounds up.") == "Rounds up."
    assert _clean("The printable quote. */") == "The printable quote."


def test_extractive_quotes_are_bounded():
    from citerag.generation import MAX_QUOTE, _candidates

    code = "\n".join(f"assert.equal(compute({i}), {i * 2}); // check doubling" for i in range(40))
    cands = _candidates(code)
    assert cands and all(20 <= len(c) <= MAX_QUOTE for c in cands)
