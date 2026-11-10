import math
from collections.abc import Sequence


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        raise ValueError("At least one relevant article ID is required.")
    return len(set(ranked_ids[:k]).intersection(relevant_ids)) / len(relevant_ids)


def mean_reciprocal_rank(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    if not relevant_ids:
        raise ValueError("At least one relevant article ID is required.")
    for rank, article_id in enumerate(ranked_ids, start=1):
        if article_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        raise ValueError("At least one relevant article ID is required.")
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, article_id in enumerate(ranked_ids[:k], start=1)
        if article_id in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / ideal_dcg


def evaluate_ranking(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(ranked_ids, relevant_ids, k),
        "mrr": mean_reciprocal_rank(ranked_ids, relevant_ids),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, relevant_ids, k),
    }
