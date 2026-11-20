import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import normalize_urdu_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.csv"
DEFAULT_FAISS = ROOT / "indexes" / "vector_index.faiss"
DEFAULT_BM25 = ROOT / "indexes" / "bm25_index.pkl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    weights: list[float] | None = None,
    rank_constant: int = 60,
) -> dict[int, float]:
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings) or any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative and match rankings")

    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        if weight == 0:
            continue
        for rank, (chunk_index, _) in enumerate(ranking, start=1):
            scores[chunk_index] = scores.get(chunk_index, 0.0) + (
                weight / (rank_constant + rank)
            )
    return scores


class Retriever:
    def __init__(
        self,
        chunks_path: str | Path = DEFAULT_CHUNKS,
        faiss_path: str | Path = DEFAULT_FAISS,
        bm25_path: str | Path = DEFAULT_BM25,
        model_name: str = "intfloat/multilingual-e5-small",
        model: Any | None = None,
        index: Any | None = None,
        bm25: Any | None = None,
    ) -> None:
        self.df = pd.read_csv(chunks_path, keep_default_na=False)
        if self.df.empty or "text" not in self.df:
            raise ValueError(f"No searchable chunks found in {chunks_path}")
        self.corpus = self.df["text"].astype(str).tolist()

        if index is None:
            index_path = Path(faiss_path)
            if not index_path.is_file():
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            try:
                import faiss
            except ImportError as error:
                message = "Search requires the 'faiss-cpu' dependency."
                raise RuntimeError(message) from error
            index = faiss.read_index(str(index_path))
        if bm25 is None:
            bm25_path = Path(bm25_path)
            if not bm25_path.is_file():
                raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
            with bm25_path.open("rb") as file:
                bm25 = pickle.load(file)
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Dense search requires the 'sentence-transformers' dependency."
                ) from error
            model = SentenceTransformer(model_name)

        self.index = index
        self.bm25 = bm25
        self.model = model
        self.model_name = model_name
        if self.index.ntotal != len(self.corpus):
            raise ValueError(
                "The FAISS index and chunk file have different numbers of rows."
            )

    def dense_search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        query_text = f"query: {query}" if "e5" in self.model_name.casefold() else query
        embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        distances, indices = self.index.search(
            np.asarray(embedding, dtype=np.float32), min(k, len(self.corpus))
        )
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], distances[0])
            if 0 <= index < len(self.corpus)
        ]

    def sparse_search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        if len(scores) != len(self.corpus):
            raise ValueError("The BM25 index and chunk file have different row counts.")
        if not np.any(scores > 0):
            return []
        ranked = np.argsort(-scores, kind="stable")[: min(k, len(self.corpus))]
        return [(int(index), float(scores[index])) for index in ranked]

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        dense_results = self.dense_search(query, k=min(len(self.corpus), k * 2))
        sparse_results = self.sparse_search(query, k=min(len(self.corpus), k * 2))
        combined_scores = reciprocal_rank_fusion(
            [dense_results, sparse_results], [alpha, 1 - alpha]
        )
        ranked = sorted(
            combined_scores.items(), key=lambda item: (-item[1], item[0])
        )[: min(k, len(self.corpus))]
        return [self._result(index, score) for index, score in ranked]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        strategy: str = "hybrid",
        alpha: float = 0.5,
    ) -> list[dict[str, Any]]:
        query = normalize_urdu_text(query.strip())
        if not query:
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")

        if strategy == "dense":
            results = self.dense_search(query, k)
        elif strategy == "bm25":
            results = self.sparse_search(query, k)
        elif strategy == "hybrid":
            return self.hybrid_search(query, k, alpha)
        else:
            raise ValueError("strategy must be 'dense', 'bm25', or 'hybrid'")
        return [self._result(index, score) for index, score in results]

    def _result(self, index: int, score: float) -> dict[str, Any]:
        row = self.df.iloc[index]
        return {
            "chunk_id": str(row["chunk_id"]),
            "score": float(score),
            "text": str(row["text"]),
            "doc_title": str(row.get("doc_title", "")),
            "url": str(row.get("url", "")),
            "lang": str(row.get("lang", "")),
            "license": str(row.get("license", "CC BY-SA 4.0")),
            "license_url": str(
                row.get(
                    "license_url",
                    "https://creativecommons.org/licenses/by-sa/4.0/",
                )
            ),
        }
