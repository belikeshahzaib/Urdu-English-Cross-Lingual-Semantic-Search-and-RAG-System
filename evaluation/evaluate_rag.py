"""Run the RAG pipeline over evaluation/rag_queries.csv and save outputs for review.

Usage (from the project root, with the venv active and indexes built):
    python evaluation/evaluate_rag.py --init-from evaluation/queries.csv --sample 20
                                                     # one-time: build rag_queries.csv
    python evaluation/evaluate_rag.py                # run all queries
    python evaluation/evaluate_rag.py --limit 3      # quick smoke test
    python evaluation/evaluate_rag.py --summarize evaluation/results/rag_outputs_<stamp>.csv

The run saves answers and citations only. It does NOT score them. Faithfulness and
answer relevance are filled in by hand in the saved CSV, following
evaluation/RAG_RUBRIC.md, and --summarize then averages the rows that were scored.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_QUERIES = ROOT / "evaluation" / "rag_queries.csv"
OUT_DIR = ROOT / "evaluation" / "results"
REQUIRED_COLUMNS = ("query_id", "query")
SCORE_COLUMNS = ("faithfulness", "answer_relevance")
OUTPUT_FIELDS = [
    "query_id", "language_type", "expected_answerable", "query", "reference_answer",
    "answer", "sufficient_evidence", "cited_chunk_ids", "cited_titles",
    "retrieved_chunk_ids", "retrieval_hit", "provider", "model", "error",
    "faithfulness", "answer_relevance", "reviewer_notes",
]


def split_ids(value: str | None) -> list[str]:
    return [part.strip() for part in re.split(r"[;|]", value or "") if part.strip()]


def _norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def load_queries(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows = [
            {k: (v or "").strip() for k, v in row.items() if k}
            for row in reader
        ]
    return [row for row in rows if row.get("query")]


INIT_FIELDS = [
    "query_id", "language_type", "expected_answerable", "query", "reference_answer",
    "relevant_article_ids", "relevant_chunk_ids", "notes",
]
UNANSWERABLE = [
    {"query_id": "na-01", "language_type": "english",
     "query": "Who won the 2050 FIFA World Cup?"},
    {"query_id": "na-02", "language_type": "urdu",
     "query": "2050 \u06a9\u0627 \u0648\u0631\u0644\u0688 \u06a9\u067e "
              "\u06a9\u0633 \u0646\u06d2 \u062c\u06cc\u062a\u0627\u061f"},
]


def init_queries(source: Path, dest: Path, sample: int = 20, force: bool = False) -> int:
    """Build rag_queries.csv from queries.csv (evenly spaced sample) plus two
    unanswerable queries. Reference answers are left blank to be filled in from
    the corpus; labels are copied as-is and are NOT verified."""
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists; use --force to overwrite it.")
    rows = load_queries(source)
    if sample and sample < len(rows):
        step = len(rows) / sample
        rows = [rows[int(i * step)] for i in range(sample)]
    out = []
    for row in rows:
        out.append({
            "query_id": row.get("query_id", ""),
            "language_type": row.get("language_type", ""),
            "expected_answerable": "yes",
            "query": row["query"],
            "reference_answer": "",
            "relevant_article_ids": row.get("relevant_article_ids")
            or row.get("relevant_articles", ""),
            "relevant_chunk_ids": row.get("relevant_chunk_ids", ""),
            "notes": "From queries.csv; labels unverified.",
        })
    for row in UNANSWERABLE:
        out.append({**{f: "" for f in INIT_FIELDS}, **row,
                    "expected_answerable": "no", "notes": "Unanswerable: should abstain."})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INIT_FIELDS)
        writer.writeheader()
        writer.writerows(out)
    return len(out)


def retrieval_hit(
    retrieved: list[dict[str, Any]],
    relevant_chunks: list[str],
    relevant_articles: list[str],
) -> str:
    """'yes'/'no' if labels exist, '' if the query has no retrieval labels."""
    if not relevant_chunks and not relevant_articles:
        return ""
    chunks = set(relevant_chunks)
    articles = {_norm(a) for a in relevant_articles}
    for passage in retrieved:
        if str(passage.get("chunk_id", "")) in chunks:
            return "yes"
        if _norm(passage.get("doc_title", "")) in articles:
            return "yes"
        if _norm(passage.get("url", "")) in articles:
            return "yes"
    return "no"


def run_query(
    row: dict[str, str],
    retriever: Any,
    generate: Callable[[str, list], Any],
    k: int,
    config_error: type[Exception],
    generation_error: type[Exception],
) -> dict[str, Any]:
    passages = retriever.retrieve(row["query"], k=k)
    record: dict[str, Any] = {
        "query_id": row.get("query_id", ""),
        "language_type": row.get("language_type", ""),
        "expected_answerable": row.get("expected_answerable", ""),
        "query": row["query"],
        "reference_answer": row.get("reference_answer", ""),
        "answer": "",
        "sufficient_evidence": "",
        "cited_chunk_ids": "",
        "cited_titles": "",
        "retrieved_chunk_ids": ";".join(str(p.get("chunk_id", "")) for p in passages),
        "retrieval_hit": retrieval_hit(
            passages,
            split_ids(row.get("relevant_chunk_ids")),
            split_ids(row.get("relevant_article_ids") or row.get("relevant_articles")),
        ),
        "provider": "",
        "model": "",
        "error": "",
        "faithfulness": "",
        "answer_relevance": "",
        "reviewer_notes": "",
        "citations": [],
        "retrieved": [
            {"chunk_id": p.get("chunk_id"), "doc_title": p.get("doc_title"),
             "url": p.get("url"), "score": p.get("score"), "text": p.get("text")}
            for p in passages
        ],
    }
    try:
        result = generate(row["query"], passages).to_dict()
    except config_error:
        raise
    except generation_error as error:
        record["error"] = str(error)
        return record
    record.update(
        answer=result["answer"],
        sufficient_evidence="yes" if result["sufficient_evidence"] else "no",
        cited_chunk_ids=";".join(c["chunk_id"] for c in result["citations"]),
        cited_titles=";".join(c["title"] for c in result["citations"]),
        provider=result["provider"],
        model=result["model"],
        citations=result["citations"],
    )
    return record


def evaluate(
    rows: list[dict[str, str]],
    retriever: Any,
    generate: Callable[[str, list], Any],
    k: int,
    config_error: type[Exception],
    generation_error: type[Exception],
    delay: float = 0.0,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    records = []
    for i, row in enumerate(rows):
        if i and delay:
            sleep(delay)  # stay under provider rate limits
        print(f"[{i + 1}/{len(rows)}] {row.get('query_id', '')} {row['query'][:60]}")
        records.append(run_query(row, retriever, generate, k, config_error, generation_error))
    return records


def save_outputs(records: list[dict[str, Any]], out_dir: Path, k: int, stamp: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"rag_outputs_{stamp}.csv"
    json_path = out_dir / f"rag_outputs_{stamp}.json"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    meta = {
        "created": stamp, "k": k, "n_queries": len(records),
        "provider": next((r["provider"] for r in records if r["provider"]), ""),
        "model": next((r["model"] for r in records if r["model"]), ""),
        "scores_recorded": False,
    }
    json_path.write_text(
        json.dumps({"meta": meta, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return csv_path, json_path


def summarize(path: Path) -> dict[str, Any]:
    """Average the hand-entered scores. Only rows with both scores count."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored = []
    for row in rows:
        values = [(row.get(c) or "").strip() for c in SCORE_COLUMNS]
        if not all(values):
            continue
        parsed = []
        for column, value in zip(SCORE_COLUMNS, values):
            if value not in ("0", "1", "2"):
                raise ValueError(
                    f"{row.get('query_id')}: {column} must be 0, 1 or 2 (got '{value}')")
            parsed.append(int(value))
        scored.append((row, *parsed))
    if not scored:
        raise ValueError("No rows have both scores filled in yet; review the outputs first.")

    def mean(values: list[int]) -> float:
        return round(sum(values) / len(values), 3)

    summary: dict[str, Any] = {
        "rows_total": len(rows),
        "rows_scored": len(scored),
        "faithfulness_mean": mean([s[1] for s in scored]),
        "answer_relevance_mean": mean([s[2] for s in scored]),
        "by_language_type": {},
    }
    for lang in sorted({s[0].get("language_type", "") for s in scored}):
        group = [s for s in scored if s[0].get("language_type", "") == lang]
        summary["by_language_type"][lang or "(blank)"] = {
            "n": len(group),
            "faithfulness_mean": mean([s[1] for s in group]),
            "answer_relevance_mean": mean([s[2] for s in group]),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N queries")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between LLM calls")
    parser.add_argument("--summarize", type=Path, help="summarize a reviewed outputs CSV")
    parser.add_argument("--init-from", type=Path, help="build rag_queries.csv from this queries.csv")
    parser.add_argument("--sample", type=int, default=20, help="rows to sample with --init-from")
    parser.add_argument("--force", action="store_true", help="overwrite rag_queries.csv")
    args = parser.parse_args(argv)

    if args.init_from:
        try:
            count = init_queries(args.init_from, args.queries, args.sample, args.force)
        except FileExistsError as error:
            print(error)
            return 1
        print(f"Wrote {count} queries to {args.queries}.\n"
              "Next: fill in reference_answer for each row from the corpus and check that "
              "relevant_article_ids match the collected articles before relying on them.")
        return 0

    if args.summarize:
        print(json.dumps(summarize(args.summarize), indent=2, ensure_ascii=False))
        return 0

    from api.app import get_retriever
    from src.generation import (
        ConfigurationError,
        GenerationError,
        generate_answer,
        load_env_file,
    )

    load_env_file(ROOT / ".env")
    rows = load_queries(args.queries)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("No queries found.")
        return 1
    try:
        records = evaluate(
            rows, get_retriever(), generate_answer, args.k,
            ConfigurationError, GenerationError, delay=args.delay)
    except ConfigurationError as error:
        print(f"Configuration problem: {error}")
        return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, json_path = save_outputs(records, OUT_DIR, args.k, stamp)
    failed = sum(1 for r in records if r["error"])
    print(f"Saved {len(records)} outputs ({failed} provider errors):\n  {csv_path}\n  {json_path}")
    print("Next: open the CSV, read each answer against its cited passages, and fill in "
          "faithfulness and answer_relevance (0-2) using evaluation/RAG_RUBRIC.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())