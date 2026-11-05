import json

import pytest
from fastapi.testclient import TestClient

from citerag.api import create_app


@pytest.fixture
def client(pipeline):
    return TestClient(create_app(pipeline))


def parse_sse(body: str) -> list[dict]:
    events = []
    for frame in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        data = json.loads(lines["data"])
        assert data["type"] == lines["event"]
        events.append(data)
    return events


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["chunks"] > 0 and body["files"] == 3
    assert body["generator"] == "extractive" and body["mode"] == "hybrid"


def test_search(client):
    body = client.get("/api/search", params={"q": "calcVat", "k": 2, "mode": "bm25"}).json()
    assert body["mode"] == "bm25" and len(body["hits"]) <= 2
    assert body["hits"][0]["title"].endswith("calcVat")


@pytest.mark.parametrize("params", [{"q": ""}, {"q": "x", "mode": "nope"}, {"q": "x", "k": 0}])
def test_search_validation(client, params):
    assert client.get("/api/search", params=params).status_code == 422


def test_ask_streams_sse(client):
    r = client.post("/api/ask", json={"question": "How are partial labor days billed?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
    n_hits = len(events[0]["hits"])
    assert all(0 <= i < n_hits for e in events if e["type"] == "cite" for i in e["hits"])


def test_ask_respects_mode(client):
    events = parse_sse(client.post("/api/ask", json={"question": "calcVat", "mode": "dense"}).text)
    assert events[0]["gate"]["mode"] == "dense"


def test_ask_abstains(client):
    events = parse_sse(client.post("/api/ask", json={"question": "Who won the 2018 world cup?"}).text)
    assert events[-1] == {"type": "done", "generator": "abstain", "abstained": True}


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001},
                                     {"question": "x", "k": 99}])
def test_ask_validation(client, payload):
    assert client.post("/api/ask", json=payload).status_code == 422


def test_generator_crash_is_reported_in_stream(client, pipeline):
    class Crash:
        name = "crash"

        def stream(self, *a):
            raise RuntimeError("boom")
            yield

    pipeline.generator = Crash()
    events = parse_sse(client.post("/api/ask", json={"question": "labor days"}).text)
    assert events[-1] == {"type": "error", "message": "internal error: RuntimeError"}


def test_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "Ask my projects" in r.text
    assert client.get("/static/app.js").status_code == 200
