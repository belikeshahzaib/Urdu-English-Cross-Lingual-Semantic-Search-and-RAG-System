import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.app import app
from evaluation.metrics import evaluate_ranking
from evaluation.run_experiments import evaluate_configuration
from src.data_collection import fetch_wikipedia_articles
from src.preprocessing import chunk_text, clean_text, create_chunks
from src.retrieval import Retriever, reciprocal_rank_fusion


class FakeIndex:
    ntotal = 3

    def search(
        self,
        embeddings: np.ndarray,
        count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if embeddings.shape[0] != 1:
            raise ValueError("Expected one query embedding.")
        indices = np.array([[0, 1, 2]], dtype=np.int64)[:, :count]
        scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)[:, :count]
        return scores, indices


class FakeBM25:
    def get_scores(self, tokens: list[str]) -> np.ndarray:
        if not tokens:
            raise ValueError("Expected query tokens.")
        return np.array([0.1, 0.9, 0.2])


class FakeModel:
    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        if not convert_to_numpy or not normalize_embeddings:
            raise ValueError("Embeddings must be normalized NumPy arrays.")
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class FakeRetriever:
    def retrieve(self, query: str, k: int) -> list[dict[str, object]]:
        return [{"query": query, "k": k}]


class FakeEvaluationRetriever:
    def retrieve(
        self,
        query: str,
        k: int,
        strategy: str,
    ) -> list[dict[str, object]]:
        if k < 1 or strategy not in {"dense", "bm25", "hybrid"}:
            raise ValueError("Invalid retrieval options.")
        return [{"doc_title": query}]


class FakeWikiPage:
    def __init__(self, title: str, links: dict[str, object] | None = None) -> None:
        self.title = title
        self.text = f"Text for {title}"
        self.fullurl = f"https://en.wikipedia.org/wiki/{title}"
        self.namespace = 0
        self.links = links or {}

    def exists(self) -> bool:
        return True


class PipelineTests(unittest.TestCase):
    def test_wikipedia_collection_respects_limit_and_attribution(self) -> None:
        linked_page = FakeWikiPage("Linked")
        seed_page = FakeWikiPage(
            "Seed",
            {"linked": linked_page, "duplicate": linked_page},
        )
        pages = {"Seed": seed_page, "Linked": linked_page}

        class FakeWiki:
            def page(self, title: str) -> FakeWikiPage:
                return pages[title]

        def create_wiki(user_agent: str, language: str) -> FakeWiki:
            if not user_agent or language != "en":
                raise ValueError("Unexpected Wikipedia client configuration.")
            return FakeWiki()

        wikipediaapi = SimpleNamespace(
            Namespace=SimpleNamespace(MAIN=0),
            Wikipedia=create_wiki,
        )
        with patch.dict("sys.modules", {"wikipediaapi": wikipediaapi}):
            articles = fetch_wikipedia_articles(["Seed"], "en", max_pages=2)
        self.assertEqual(articles["title"].tolist(), ["Seed", "Linked"])
        self.assertEqual(set(articles["license"]), {"CC BY-SA 4.0"})
        self.assertTrue(
            articles["license_url"].str.contains("creativecommons.org").all()
        )

    def test_urdu_normalization_and_chunk_overlap(self) -> None:
        self.assertEqual(clean_text("ك\u064eت\u064eب\u064e"), "کتب")
        self.assertEqual(
            chunk_text("one two three four five", chunk_size=3, overlap=1),
            ["one two three", "three four five"],
        )

    def test_chunk_records_include_language_and_license(self) -> None:
        corpus = pd.DataFrame(
            [
                {
                    "title": "پاکستان",
                    "text": "ک\u064eتاب اور تاریخ",
                    "url": "https://ur.wikipedia.org/wiki/پاکستان",
                    "lang": "ur",
                }
            ]
        )
        chunks = create_chunks(corpus, chunk_size=2, overlap=0)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks.loc[0, "chunk_id"].startswith("ur:پاکستان:"))
        self.assertEqual(chunks.loc[0, "license"], "CC BY-SA 4.0")
        self.assertEqual(
            chunks.loc[0, "license_url"],
            "https://creativecommons.org/licenses/by-sa/4.0/",
        )

    def test_rrf_and_retriever_return_ranked_metadata(self) -> None:
        fused_scores = reciprocal_rank_fusion(
            [[(0, 0.1), (1, 0.2)], [(0, 8.0), (2, 1.0)]],
            [0.5, 0.5],
        )
        self.assertGreater(fused_scores[0], fused_scores[1])
        self.assertEqual(reciprocal_rank_fusion([[(1, 1.0)]], [0.0]), {})
        frame = pd.DataFrame(
            [
                {
                    "chunk_id": f"id-{index}",
                    "text": f"passage {index}",
                    "doc_title": f"article-{index}",
                    "url": f"https://example.test/{index}",
                    "lang": "en",
                }
                for index in range(3)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            chunks_path = Path(directory) / "chunks.csv"
            frame.to_csv(chunks_path, index=False)
            retriever = Retriever(
                chunks_path,
                model_name="multilingual-minilm",
                model=FakeModel(),
                index=FakeIndex(),
                bm25=FakeBM25(),
            )
            result = retriever.retrieve("example query", k=2, strategy="bm25")
        self.assertEqual(result[0]["chunk_id"], "id-1")
        self.assertEqual(result[0]["doc_title"], "article-1")
        self.assertEqual(result[0]["lang"], "en")
        self.assertEqual(
            retriever.retrieve("!!!", k=2, strategy="bm25"),
            [],
        )
        hybrid_result = retriever.retrieve("example query", k=2)
        self.assertEqual(hybrid_result[0]["chunk_id"], "id-1")

    def test_evaluation_metrics(self) -> None:
        metrics = evaluate_ranking(
            ["unrelated", "target"], {"target"}, k=2
        )
        self.assertEqual(metrics["recall@2"], 1.0)
        self.assertEqual(metrics["mrr"], 0.5)
        self.assertEqual(metrics["ndcg@2"], 1 / np.log2(3))

    def test_evaluation_reports_query_types_separately(self) -> None:
        queries = pd.DataFrame(
            [
                {
                    "query_id": "en-01",
                    "query": "English target",
                    "language_type": "english",
                    "relevant_article_ids": "English target",
                },
                {
                    "query_id": "xu-01",
                    "query": "Urdu target",
                    "language_type": "english_to_urdu",
                    "relevant_article_ids": "Urdu target",
                },
            ]
        )
        rows = evaluate_configuration(
            FakeEvaluationRetriever(),
            queries,
            ("dense", "bm25", "hybrid"),
            1,
            "e5",
            200,
        )
        self.assertEqual(len(rows), 9)
        overall = next(row for row in rows if row["language_type"] == "all")
        self.assertEqual(overall["query_count"], 2)
        self.assertEqual(overall["recall@1"], 1.0)

    def test_search_api_validation_and_results(self) -> None:
        client = TestClient(app)
        with patch("api.app.get_retriever", return_value=FakeRetriever()):
            response = client.post(
                "/search", json={"query": "  Pakistan  ", "k": 3}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"results": [{"query": "Pakistan", "k": 3}]},
        )
        self.assertEqual(
            client.post("/search", json={"query": "  "}).status_code,
            422,
        )
        self.assertEqual(
            client.post("/search", json={"query": "Pakistan", "k": 51}).status_code,
            422,
        )
        self.assertEqual(client.get("/").status_code, 200)

    def test_search_api_reports_missing_index(self) -> None:
        client = TestClient(app)
        with patch(
            "api.app.get_retriever",
            side_effect=FileNotFoundError("FAISS index unavailable"),
        ):
            response = client.post("/search", json={"query": "Pakistan"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "FAISS index unavailable")


if __name__ == "__main__":
    unittest.main()
