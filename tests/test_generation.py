"""Generation tests. All provider calls are mocked; no API key is needed."""
from unittest.mock import patch

import pytest

from src import generation as g

PASSAGES = [
    {"chunk_id": "c1", "title": "Lahore", "url": "https://en.wikipedia.org/wiki/Lahore",
     "text": "Lahore is the capital of Punjab."},
    {"chunk_id": "c2", "title": "Karachi", "url": "https://en.wikipedia.org/wiki/Karachi",
     "text": "Karachi is the capital of Sindh."},
]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")


def test_prompt_numbers_passages_and_contains_query():
    prompt = g.build_prompt("What is Lahore?", g.normalize_passages(PASSAGES))
    assert "[1] Title: Lahore" in prompt and "[2] Title: Karachi" in prompt
    assert "What is Lahore?" in prompt


def test_normalize_accepts_alternate_field_names_and_skips_empty():
    out = g.normalize_passages([
        {"id": "x", "article_title": "T", "source_url": "u", "chunk_text": "hello"},
        {"chunk_id": "empty", "text": "   "},
    ])
    assert out == [{"chunk_id": "x", "title": "T", "url": "u", "text": "hello",
                    "license": "", "license_url": ""}]


def test_normalize_uses_retriever_field_names():
    out = g.normalize_passages([{
        "chunk_id": "c9", "doc_title": "Indus", "url": "https://x", "text": "river",
        "license": "CC BY-SA 4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "score": 0.5, "lang": "en",
    }])
    assert out[0]["title"] == "Indus"
    assert out[0]["license"] == "CC BY-SA 4.0"


def test_normalize_truncates_long_text():
    out = g.normalize_passages([{"chunk_id": "a", "text": "x" * 5000}])
    assert len(out[0]["text"]) == g.MAX_PASSAGE_CHARS


def test_answer_with_citations():
    with patch.object(g, "_call_provider", return_value="Lahore is in Punjab [1]."):
        r = g.generate_answer("Where is Lahore?", PASSAGES)
    assert r.sufficient_evidence
    assert [c.chunk_id for c in r.citations] == ["c1"]
    assert r.citations[0].url.endswith("/Lahore")
    assert r.provider == "groq" and r.model == "test-model"


def test_invented_markers_are_dropped_and_duplicates_collapsed():
    with patch.object(g, "_call_provider", return_value="A [2] B [2] C [9] D [0]."):
        r = g.generate_answer("q", PASSAGES)
    assert [c.marker for c in r.citations] == [2]


def test_insufficient_evidence_from_model():
    with patch.object(g, "_call_provider", return_value=g.INSUFFICIENT_TOKEN):
        r = g.generate_answer("Who won the 2050 cup?", PASSAGES)
    assert not r.sufficient_evidence and r.citations == []
    assert r.answer == g.INSUFFICIENT_MESSAGE


def test_no_passages_skips_provider_call():
    with patch.object(g, "_call_provider") as m:
        r = g.generate_answer("q", [])
    m.assert_not_called()
    assert not r.sufficient_evidence


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        g.generate_answer("   ", PASSAGES)


@pytest.mark.parametrize("var", ["LLM_MODEL", "GROQ_API_KEY"])
def test_missing_config_raises(monkeypatch, var):
    monkeypatch.delenv(var)
    with pytest.raises(g.ConfigurationError):
        g.generate_answer("q", PASSAGES)


def test_unsupported_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(g.ConfigurationError):
        g.generate_answer("q", PASSAGES)


def test_provider_failure_propagates():
    with (
        patch.object(g, "_call_provider", side_effect=g.ProviderError("boom")),
        pytest.raises(g.ProviderError),
    ):
        g.generate_answer("q", PASSAGES)


def test_empty_provider_answer_is_error():
    with (
        patch.object(g, "_call_provider", return_value="  "),
        pytest.raises(g.ProviderError),
    ):
        g.generate_answer("q", PASSAGES)


def test_to_dict_shape():
    with patch.object(g, "_call_provider", return_value="ok [1]"):
        d = g.generate_answer("q", PASSAGES).to_dict()
    assert set(d) == {"query", "answer", "sufficient_evidence", "citations", "provider", "model"}
    assert set(d["citations"][0]) == {
        "marker", "chunk_id", "title", "url", "text", "license", "license_url"}


def test_load_env_file_sets_missing_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nFOO_KEY="abc"\nEXISTING=new\n', encoding="utf-8")
    monkeypatch.setenv("EXISTING", "keep")
    monkeypatch.delenv("FOO_KEY", raising=False)
    g.load_env_file(env)
    import os
    assert os.environ["FOO_KEY"] == "abc"
    assert os.environ["EXISTING"] == "keep"
    monkeypatch.delenv("FOO_KEY", raising=False)


def test_load_env_file_missing_file_is_ignored(tmp_path):
    g.load_env_file(tmp_path / "nope.env")


def test_post_json_sends_explicit_user_agent():
    import io
    import json as _json

    seen = {}

    class FakeResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["ua"] = req.get_header("User-agent")
        return FakeResp(_json.dumps({"ok": True}).encode())

    with patch.object(g.urllib.request, "urlopen", fake_urlopen):
        assert g._post_json("https://example.com", {"X": "1"}, {"a": 1}) == {"ok": True}
    assert seen["ua"] == g.USER_AGENT
    assert "urllib" not in seen["ua"].lower()


def test_post_json_http_error_includes_status_and_body():
    import io
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"error":"blocked"}'))

    with (
        patch.object(g.urllib.request, "urlopen", fake_urlopen),
        pytest.raises(g.ProviderError) as exc,
    ):
        g._post_json("https://example.com", {}, {})
    assert "403" in str(exc.value) and "blocked" in str(exc.value)