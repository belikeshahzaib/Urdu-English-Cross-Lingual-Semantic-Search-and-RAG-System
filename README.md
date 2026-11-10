# Urdu-English Cross-Lingual Semantic Search

A bilingual Wikipedia passage collection, preprocessing, retrieval, evaluation, and search application for English and Urdu. The current implementation covers Shahzaib's data and retrieval tasks; it does not implement the separate RAG answer-generation workflow.

## Setup

Use Python 3.10 or later. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The embedding models are downloaded from Hugging Face when an index is built and can be run on CPU. The first model run requires an internet connection and may take some time.

## Build the corpus and indexes

The collector starts from Pakistan-related English and Urdu Wikipedia topics and follows article links, up to 5,000 pages per language by default:

```powershell
python -m src.data_collection
```

Use `--max-pages-per-language` to choose a smaller pilot size or a different collection limit. The collector saves article text, page URLs, language, and the CC BY-SA 4.0 attribution fields to `data/raw/corpus.csv`.

Clean, normalize and split the corpus into overlapping word-count chunks:

```powershell
python -m src.preprocessing --chunk-size 200 --overlap 50
```

Build a dense FAISS index and a BM25 index:

```powershell
python -m src.indexing --model e5
```

The three model choices are `e5`, `bge-m3`, and `minilm`. They are stored separately under `indexes/e5`, `indexes/bge-m3`, and `indexes/minilm`. Choose another `--index-dir` to store a chunk-size or model experiment without replacing an existing index. E5 uses its required `query:` and `passage:` prefixes. Dense vectors are normalized and searched with inner product.

The pipeline writes its canonical files under `data/raw`, `data/processed`, and `indexes`. Those generated files are excluded from Git; rerun the commands after cloning.

## Retrieval interface

```python
from src.retrieval import Retriever

retriever = Retriever(
    "data/processed/chunks.csv",
    "indexes/e5/vector_index.faiss",
    "indexes/e5/bm25_index.pkl",
    "intfloat/multilingual-e5-small",
)
results = retriever.retrieve("What is the Indus River?", k=5)
```

`retrieve(query, k)` returns ranked dictionaries with `chunk_id`, `score`, `text`, `doc_title`, `url`, `lang`, `license`, and `license_url`. The default `strategy="hybrid"` fuses dense and BM25 rankings using weighted reciprocal-rank fusion, avoiding direct addition of scores that have different scales. Use `strategy="dense"` or `"bm25"` for the baselines. Scores are strategy-specific and should not be compared across strategies as calibrated probabilities.

## Search API and interface

Build the default E5 index first, then start the API from the repository root:

```powershell
uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000/` for the search page. `POST /search` accepts `{"query": "...", "k": 5}` and returns ranked passages and source URLs. `/health` reports whether the corpus and index files exist. The first successful search loads the retrieval model; missing files or dependencies return a clear service-unavailable response.

## Evaluation

`evaluation/queries.csv` provides 40 hand-written English queries: 20 English-to-English and 20 English-to-Urdu queries. The `relevant_article_ids` values identify Wikipedia article titles, and multiple labels may be separated with `|`. Review these labels against the collected article titles before treating metrics as final ground truth.

Run the model, retrieval-strategy, and chunk-size comparisons:

```powershell
python -m evaluation.run_experiments
```

The default experiment compares E5, BGE-M3, and multilingual MiniLM; chunk sizes 100, 200, and 300 words; and dense, BM25, and hybrid retrieval, using 50-word overlap. It reports macro Recall@10, MRR, and nDCG@10 by query type and overall to `evaluation/results.csv`. To run a smaller pilot:

```powershell
python -m evaluation.run_experiments --models e5 --chunk-sizes 200 --k 5
```

Experiment runs require a collected corpus and may download models. No benchmark results are included in advance; use the generated table from a completed run rather than reporting unmeasured values.

## Wikipedia attribution

Wikipedia article text is reused under CC BY-SA 4.0. The corpus preserves each article's title, source URL, language, and license; search results link back to the source. Keep these attribution fields when distributing processed data and include attribution and the license when redistributing adapted text.
