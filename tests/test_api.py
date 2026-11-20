"""API tests for /search and /answer. No live retriever or API key needed."""
import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from src import generation as g

PASSAGES = [
    {"chunk_id": "c1", "score": 0.9, "text": "Lahore is the capital of Punjab.",
     "doc_title": "Lahore", "url": "https://en.wikipedia.org/wiki/Lahore", "lang": "en",
     "license": "CC BY-SA 4.0",
     "license_url": "https://creativecommons.org/licenses/by-sa/4.0/"},
]


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = PASSAGES if results is None else results
        self.error = error
        self.calls = []

    def retrieve(self, query, k=5, **kwargs):
        self.calls.append((query, k))
        if self.error:
            raise self.error
        return self.results


@pytest.fixture
def client(monkeypatch):
    fake = FakeRetriever()
    monkeypatch.setattr(app_module, "get_retriever", lambda: fake)
    c = TestClient(app_module.app)
    c.fake = fake
    return c


def ok_result(query="q"):
    return g.AnswerResult(
        query, "Lahore is in Punjab [1].", True,
        [g.Citation(1, "c1", "Lahore", "https://en.wikipedia.org/wiki/Lahore",
                    "Lahore is the capital of Punjab.", "CC BY-SA 4.0",
                    "https://creativecommons.org/licenses/by-sa/4.0/")],
        "groq", "test-model",
    )


def test_search_still_works(client):
    r = client.post("/search", json={"query": "Lahore", "k": 3})
    assert r.status_code == 200
    assert r.json() == {"results": PASSAGES}
    assert client.fake.calls == [("Lahore", 3)]


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": "   "}, {"query": "x", "k": 0}, {"query": "x", "k": 51},
    {"query": "x" * 2001},
])
def test_answer_validation_rejects_bad_requests(client, body):
    assert client.post("/answer", json=body).status_code == 422


def test_answer_success_shape(client, monkeypatch):
    seen = {}

    def fake_generate(query, passages):
        seen["query"], seen["passages"] = query, passages
        return ok_result(query)

    monkeypatch.setattr(app_module, "generate_answer", fake_generate)
    r = client.post("/answer", json={"query": "  Where is Lahore?  ", "k": 4})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"query", "answer", "sufficient_evidence", "citations",
                         "provider", "model"}
    assert body["citations"][0]["chunk_id"] == "c1"
    assert body["citations"][0]["url"].startswith("https://")
    assert client.fake.calls == [("Where is Lahore?", 4)]
    assert seen["passages"] == PASSAGES


def test_answer_insufficient_evidence_is_200(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "generate_answer",
        lambda q, p: g.AnswerResult(q, g.INSUFFICIENT_MESSAGE, False, [], "groq", "m"),
    )
    r = client.post("/answer", json={"query": "Who wins in 2050?"})
    assert r.status_code == 200
    assert r.json()["sufficient_evidence"] is False
    assert r.json()["citations"] == []


def test_answer_missing_configuration_returns_503(client, monkeypatch):
    def boom(q, p):
        raise g.ConfigurationError("LLM_MODEL is not set.")

    monkeypatch.setattr(app_module, "generate_answer", boom)
    r = client.post("/answer", json={"query": "x"})
    assert r.status_code == 503
    assert "LLM_MODEL" in r.json()["detail"]


def test_answer_provider_failure_returns_502(client, monkeypatch):
    def boom(q, p):
        raise g.ProviderError("Provider returned HTTP 429.")

    monkeypatch.setattr(app_module, "generate_answer", boom)
    r = client.post("/answer", json={"query": "x"})
    assert r.status_code == 502


@pytest.mark.parametrize("error", [FileNotFoundError("index missing"), RuntimeError("no faiss")])
def test_answer_retriever_unavailable_returns_503(monkeypatch, error):
    monkeypatch.setattr(app_module, "get_retriever", lambda: FakeRetriever(error=error))
    r = TestClient(app_module.app).post("/answer", json={"query": "x"})
    assert r.status_code == 503


def test_answer_does_not_leak_api_key(client, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-key")

    def boom(q, p):
        raise g.ProviderError("Provider returned HTTP 401.")

    monkeypatch.setattr(app_module, "generate_answer", boom)
    r = client.post("/answer", json={"query": "x"})
    assert "super-secret-key" not in r.text