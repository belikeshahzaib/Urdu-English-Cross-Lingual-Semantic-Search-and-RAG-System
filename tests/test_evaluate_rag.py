"""Tests for evaluation/evaluate_rag.py helpers. No retriever or API key needed."""
import csv

import pytest

from evaluation import evaluate_rag as er


class Err(Exception):
    pass


class ConfigErr(Exception):
    pass


class FakeResult:
    def __init__(self, answer="A [1].", ok=True):
        self.d = {
            "query": "q", "answer": answer, "sufficient_evidence": ok,
            "citations": [{"marker": 1, "chunk_id": "c1", "title": "Lahore", "url": "u",
                           "text": "t", "license": "", "license_url": ""}] if ok else [],
            "provider": "groq", "model": "m",
        }

    def to_dict(self):
        return self.d


class FakeRetriever:
    def retrieve(self, query, k=5):
        return [{"chunk_id": "c1", "doc_title": "Lahore", "url": "https://x/Lahore",
                 "score": 0.9, "text": "Lahore is in Punjab."}]


def test_split_ids():
    assert er.split_ids("a; b|c ;") == ["a", "b", "c"]
    assert er.split_ids("") == [] and er.split_ids(None) == []


def test_retrieval_hit_cases():
    got = FakeRetriever().retrieve("q")
    assert er.retrieval_hit(got, ["c1"], []) == "yes"
    assert er.retrieval_hit(got, [], ["lahore"]) == "yes"
    assert er.retrieval_hit(got, [], ["https://x/Lahore"]) == "yes"
    assert er.retrieval_hit(got, ["zzz"], ["Karachi"]) == "no"
    assert er.retrieval_hit(got, [], []) == ""


def test_load_queries_requires_columns(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        er.load_queries(p)


def test_load_queries_reads_utf8_bom_and_skips_blank(tmp_path):
    p = tmp_path / "q.csv"
    p.write_text("query_id,query\nq1,لاہور کہاں ہے؟\nq2,\n", encoding="utf-8-sig")
    rows = er.load_queries(p)
    assert len(rows) == 1 and rows[0]["query"].startswith("لاہور")


def test_evaluate_records_answers_and_errors():
    rows = [{"query_id": "q1", "query": "a", "relevant_articles": "Lahore"},
            {"query_id": "q2", "query": "b"}]
    calls = []

    def gen(query, passages):
        calls.append(query)
        if query == "b":
            raise Err("HTTP 429")
        return FakeResult()

    sleeps = []
    recs = er.evaluate(rows, FakeRetriever(), gen, 5, ConfigErr, Err, delay=1.5,
                       sleep=sleeps.append)
    assert recs[0]["answer"] == "A [1]." and recs[0]["cited_chunk_ids"] == "c1"
    assert recs[0]["retrieval_hit"] == "yes"
    assert recs[1]["error"] == "HTTP 429" and recs[1]["answer"] == ""
    assert sleeps == [1.5]


def test_config_error_aborts_run():
    def gen(query, passages):
        raise ConfigErr("no key")

    with pytest.raises(ConfigErr):
        er.evaluate([{"query_id": "q", "query": "a"}], FakeRetriever(), gen, 5, ConfigErr, Err)


def test_save_outputs_leaves_scores_blank(tmp_path):
    rec = er.run_query({"query_id": "q1", "query": "a"}, FakeRetriever(),
                       lambda q, p: FakeResult(), 5, ConfigErr, Err)
    csv_path, json_path = er.save_outputs([rec], tmp_path, 5, "t")
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["faithfulness"] == "" and row["answer_relevance"] == ""
    assert '"scores_recorded": false' in json_path.read_text(encoding="utf-8")


def _write_reviewed(tmp_path, rows):
    p = tmp_path / "r.csv"
    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=er.OUTPUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in er.OUTPUT_FIELDS})
    return p


def test_summarize_counts_only_scored_rows(tmp_path):
    p = _write_reviewed(tmp_path, [
        {"query_id": "q1", "language_type": "en", "faithfulness": "2", "answer_relevance": "1"},
        {"query_id": "q2", "language_type": "ur", "faithfulness": "0", "answer_relevance": "1"},
        {"query_id": "q3", "language_type": "ur"},
    ])
    s = er.summarize(p)
    assert s["rows_total"] == 3 and s["rows_scored"] == 2
    assert s["faithfulness_mean"] == 1.0 and s["answer_relevance_mean"] == 1.0
    assert s["by_language_type"]["ur"]["n"] == 1


def test_summarize_refuses_when_nothing_scored(tmp_path):
    p = _write_reviewed(tmp_path, [{"query_id": "q1"}])
    with pytest.raises(ValueError):
        er.summarize(p)


def test_summarize_rejects_out_of_range_score(tmp_path):
    p = _write_reviewed(tmp_path, [{"query_id": "q1", "faithfulness": "5", "answer_relevance": "1"}])
    with pytest.raises(ValueError):
        er.summarize(p)