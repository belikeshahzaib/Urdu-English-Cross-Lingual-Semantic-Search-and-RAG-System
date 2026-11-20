import argparse
from pathlib import Path

import pandas as pd

from evaluation.metrics import evaluate_ranking
from src.indexing import MODEL_NAMES, build_indexes
from src.preprocessing import preprocess_corpus
from src.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "evaluation" / "queries.csv"
DEFAULT_CORPUS = ROOT / "data" / "raw" / "corpus.csv"
DEFAULT_WORK_DIR = ROOT / "evaluation" / ".work"
DEFAULT_RESULTS = ROOT / "evaluation" / "results.csv"
STRATEGIES = ("dense", "bm25", "hybrid")


def parse_int_list(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        message = "Provide comma-separated positive integers."
        raise argparse.ArgumentTypeError(message) from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    return values


def evaluate_configuration(
    retriever: Retriever,
    queries: pd.DataFrame,
    strategies: tuple[str, ...],
    k: int,
    model: str,
    chunk_size: int,
) -> list[dict[str, object]]:
    language_types = sorted(set(queries["language_type"])) + ["all"]
    scores: dict[str, dict[str, list[dict[str, float]]]] = {
        strategy: {language_type: [] for language_type in language_types}
        for strategy in strategies
    }
    for query in queries.itertuples(index=False):
        relevant = {
            article_id.strip()
            for article_id in query.relevant_article_ids.split("|")
            if article_id.strip()
        }
        if not relevant:
            raise ValueError(f"Query {query.query_id} has no relevant article IDs.")
        for strategy in strategies:
            results = retriever.retrieve(query.query, k=k, strategy=strategy)
            article_ids = [result["doc_title"] for result in results]
            ranking_scores = evaluate_ranking(article_ids, relevant, k)
            scores[strategy][query.language_type].append(ranking_scores)
            scores[strategy]["all"].append(ranking_scores)

    return [
        {
            "model": model,
            "chunk_size": chunk_size,
            "retrieval_strategy": strategy,
            "language_type": language_type,
            "query_count": len(rows),
            f"recall@{k}": sum(row[f"recall@{k}"] for row in rows) / len(rows),
            "mrr": sum(row["mrr"] for row in rows) / len(rows),
            f"ndcg@{k}": sum(row[f"ndcg@{k}"] for row in rows) / len(rows),
        }
        for strategy, type_scores in scores.items()
        for language_type, rows in type_scores.items()
        if rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare embedding models, retrieval strategies, and chunk sizes."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--models",
        default="e5,bge-m3,minilm",
        help="Comma-separated model keys: e5, bge-m3, minilm.",
    )
    parser.add_argument("--chunk-sizes", type=parse_int_list, default=[100, 200, 300])
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    model_keys = [item.strip() for item in args.models.split(",") if item.strip()]
    invalid_models = sorted(set(model_keys).difference(MODEL_NAMES))
    if invalid_models or not model_keys:
        parser.error(f"Unknown or empty model list: {', '.join(invalid_models)}")
    if args.overlap < 0 or any(args.overlap >= size for size in args.chunk_sizes):
        parser.error("Overlap must be non-negative and smaller than each chunk size.")
    if args.k < 1:
        parser.error("k must be at least 1.")
    if not args.corpus.is_file():
        parser.error(f"Corpus not found: {args.corpus}")
    if not args.queries.is_file():
        parser.error(f"Evaluation queries not found: {args.queries}")

    queries = pd.read_csv(args.queries, keep_default_na=False)
    required = {"query_id", "query", "language_type", "relevant_article_ids"}
    missing = required.difference(queries.columns)
    if missing or queries.empty:
        parser.error(
            f"Evaluation query file is empty or missing columns: {sorted(missing)}"
        )
    try:
        corpus_titles = set(
            pd.read_csv(
                args.corpus,
                usecols=["title"],
                keep_default_na=False,
            )["title"]
        )
    except pd.errors.EmptyDataError:
        parser.error(f"Corpus file is empty: {args.corpus}")
    except ValueError as error:
        parser.error(f"Corpus file does not contain article titles: {error}")
    labeled_titles = {
        article_id.strip()
        for labels in queries["relevant_article_ids"]
        for article_id in labels.split("|")
        if article_id.strip()
    }
    missing_titles = sorted(labeled_titles.difference(corpus_titles))
    if missing_titles:
        parser.error(
            "Evaluation labels must match article titles in the collected corpus. "
            f"Missing titles: {', '.join(missing_titles)}"
        )

    output_rows: list[dict[str, object]] = []
    for chunk_size in args.chunk_sizes:
        chunks_path = args.work_dir / f"chunks-{chunk_size}.csv"
        preprocess_corpus(args.corpus, chunks_path, chunk_size, args.overlap)
        for model_key in model_keys:
            model_name = MODEL_NAMES[model_key]
            index_dir = args.work_dir / "indexes" / f"{model_key}-chunk-{chunk_size}"
            dense_path, sparse_path = build_indexes(
                chunks_path, index_dir, model_name
            )
            retriever = Retriever(
                chunks_path, dense_path, sparse_path, model_name
            )
            output_rows.extend(
                evaluate_configuration(
                    retriever,
                    queries,
                    STRATEGIES,
                    args.k,
                    model_key,
                    chunk_size,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(args.output, index=False)
    print(f"Saved {len(output_rows)} experiment rows to {args.output}")


if __name__ == "__main__":
    main()
