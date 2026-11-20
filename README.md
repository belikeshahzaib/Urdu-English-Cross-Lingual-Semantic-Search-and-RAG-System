# Urdu-English Cross-Lingual Semantic Search and RAG System

A bilingual Wikipedia passage collection, preprocessing, retrieval, evaluation and search application for English and Urdu, plus retrieval-augmented answer generation (RAG) that answers questions from the retrieved passages with citations.

## Setup

Use Python 3.10 or later. From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The embedding models are downloaded from Hugging Face when an index is built and can be run on CPU. The first model run requires an internet connection and may take some time.

## Project structure

```text
├── api/
│   └── app.py                  # FastAPI service: /search, /answer, /health, web interface
├── data/
│   ├── raw/                    # Collected Wikipedia corpus (corpus.csv)
│   └── processed/              # Cleaned and chunked passages (chunks.csv)
├── evaluation/
│   ├── queries.csv             # 40 retrieval test queries
│   ├── run_experiments.py      # Model, retrieval-strategy, and chunk-size benchmarks
│   ├── metrics.py              # Recall@k, MRR, nDCG@k
│   ├── results.csv             # Generated retrieval metrics
│   ├── rag_queries.csv         # 26 RAG queries with reference answers and labels
│   ├── evaluate_rag.py         # Runs the RAG pipeline and saves answers for review
│   ├── RAG_RUBRIC.md           # Manual scoring rubric (faithfulness, answer relevance)
│   └── results/                # Saved RAG outputs (CSV and JSON)
├── frontend/
│   └── index.html              # Search page with "Get answer" button
├── indexes/
│   ├── e5/                     # FAISS and BM25 indexes for multilingual-e5
│   ├── bge-m3/                 # FAISS and BM25 indexes for bge-m3
│   └── minilm/                 # FAISS and BM25 indexes for multilingual-minilm
├── src/
│   ├── data_collection.py      # Bilingual Wikipedia crawler and attribution collector
│   ├── preprocessing.py        # Text cleaning, normalization, and word-count chunking
│   ├── indexing.py             # Dense FAISS and lexical BM25 index builder
│   ├── retrieval.py            # Hybrid reciprocal-rank fusion, dense, and BM25 retriever
│   └── generation.py           # RAG prompt, provider calls, citation parsing
├── tests/                      # pytest suite (mocked provider, no API key needed)
├── pytest.ini
└── requirements.txt
```

## Build the corpus and indexes

The collector starts from Pakistan-related English and Urdu Wikipedia topics and follows article links, up to 5,000 pages per language by default:

```powershell
python -m src.data_collection
```

Use `--max-pages-per-language` to choose a smaller pilot size or a different collection limit. The collector saves article text, page URLs, language, and the CC BY-SA 4.0 attribution fields to `data/raw/corpus.csv`.

Clean, normalize, and split the corpus into overlapping word-count chunks:

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

## RAG answer generation

`POST /answer` retrieves passages with the existing `Retriever`, then asks an LLM to answer using only those passages. Provider calls are isolated in `src/generation.py`, so the API route never talks to a provider directly and tests can mock it.

### Provider setup

The project uses **Groq** with the model **`openai/gpt-oss-120b`** (an open-weight model hosted by Groq). Gemini is also supported by the code.

Create a local `.env` file in the repository root. It is git-ignored; never commit it or paste keys into source files, issues or chat.

```text
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=your-key-here
```

To use Gemini instead, set `LLM_PROVIDER=gemini`, `LLM_MODEL` to a Gemini model name, and `GEMINI_API_KEY`. No extra packages are needed; the provider is called with the standard library.

### Using /answer

Start the server (see above), then either use the **Get answer** button on the web page or call the API:

```powershell
curl -X POST http://127.0.0.1:8000/answer -H "Content-Type: application/json" -d "{\"query\": \"What is the capital of Pakistan?\", \"k\": 5}"
```

The response contains:

| Field | Meaning |
|---|---|
| `answer` | The generated answer, with `[n]` citation markers |
| `sufficient_evidence` | `false` if the passages do not support an answer |
| `citations` | For each cited passage: marker, chunk ID, title, URL, text, license |
| `provider`, `model` | What produced the answer |

When the passages do not contain enough information, the model is instructed to abstain: `sufficient_evidence` is `false`, a fixed insufficient-evidence message is returned (currently in English for all queries), and no citations are returned.

Errors: `422` invalid request, `503` missing configuration or unavailable index, `502` provider failure.

The page renders the query and model output as text (never as HTML). The model may write markdown such as `**bold**`, which then appears as literal characters.

### Limitations

- **Rate limits.** Free-tier limits apply and may change; check your provider console for current limits. On the Groq on-demand tier, `openai/gpt-oss-120b` returned HTTP 429 errors at a limit of 8,000 tokens per minute during evaluation.
- **Answers can still be wrong.** The model is told to use only the passages, but the code does not enforce it. Check the cited passages before relying on an answer.
- **Retrieval limits answer quality.** The model only sees the top `k` passages. If retrieval misses the right passage, the answer can be incomplete or the model may abstain.
- **Cost.** Free-tier use costs nothing, but paid use is billed by the provider per token.

## Retrieval evaluation

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

## RAG evaluation

`evaluation/rag_queries.csv` has 26 queries: 24 answerable (English, English-to-Urdu and Urdu) and 2 unanswerable (`na-01`, `na-02`) that should be answered with the insufficient-evidence reply. Reference answers and article/chunk labels were checked against the collected corpus.

Run it from the repository root with the virtual environment active, indexes built, and `.env` configured:

```powershell
python evaluation/evaluate_rag.py --limit 3          # smoke test
python evaluation/evaluate_rag.py --delay 30         # full run
python evaluation/evaluate_rag.py --summarize evaluation/results/rag_outputs_<stamp>.csv
```

Use `--delay 30` (seconds between calls) on the Groq free tier to stay under the token-per-minute limit; with the default 2 seconds, most queries failed with HTTP 429. Use `--k` to change the number of retrieved passages (default 5).

The script saves answers, citations and retrieved chunk IDs to `evaluation/results/rag_outputs_<stamp>.csv` and `.json`. It does **not** score them. Faithfulness and answer relevance (0 to 2) are entered by hand following `evaluation/RAG_RUBRIC.md`, scored against the retrieved passages and not against outside knowledge. A second reviewer independently scores at least 10 rows. `--summarize` averages only rows where both scores are filled in. Rows with a provider error are left unscored and re-run.

Notes for interpreting the output:

- `retrieval_hit` is an exact match on article title or chunk ID. For English-to-Urdu queries the labels are Urdu article titles, so an English article on the same topic counts as a miss even when it answers the question. Treat `retrieval_hit` as a retrieval diagnostic, separate from generation quality.
- Scores come from a small query set and a small number of reviewers, so they show how the system behaved on these queries and should not be read as general accuracy.

### Results

RAG quality scores will be added here after all outputs have been reviewed by two scorers. Report the provider, model, `k`, number of queries and reviewer setup alongside any numbers.

## Tests and lint

```powershell
pytest -v
ruff check .
```

The tests use mocked provider responses and need no API key. `pytest.ini` puts the repository root on the import path.

## Wikipedia attribution

Wikipedia article text is reused under CC BY-SA 4.0. The corpus preserves each article's title, source URL, language, and license; search results and answer citations link back to the source. Keep these attribution fields when distributing processed data and include attribution and the license when redistributing adapted text.

## Future work

**Evaluation**
- Finish the RAG review: a second reviewer independently scores at least 10 rows, agreement is recorded, and the final scores are added to the Results section.
- Re-check faithfulness on rows where retrieval missed the labeled article, to see whether the model answered from outside knowledge instead of the cited passage.
- Decide how a correct abstention in the wrong language is scored, and state it in `RAG_RUBRIC.md`.
- Grow `rag_queries.csv` beyond 26 queries, with more Urdu-language questions, multi-passage questions and unanswerable ones, so per-language scores rest on more than a handful of rows.
- Extend the English-to-Urdu labels to include the English article title, or add a cross-language match, so `retrieval_hit` reflects relevant content and not only exact titles.

**Generation**
- Strengthen the prompt, or add a check that the cited passage supports each claim, to reduce answers that come from model memory.
- Return the insufficient-evidence message in the language of the question (it is currently English only).
- Ask the model for plain text, or render markdown safely, so `**bold**` does not appear as literal asterisks.
- Add a retry with backoff for HTTP 429 rate-limit errors, so evaluation no longer needs a long fixed `--delay`.
- Compare other models or reasoning settings for speed and quality. Any change means re-running and re-scoring the evaluation, so scores from different configurations are not mixed.

**Retrieval (coordinate with Shahzaib)**
- Tune `k` and the hybrid weight `alpha` for answer quality, not only for ranking metrics.
- Try reranking the top passages before generation, and test whether cross-language retrieval improves for English queries about Urdu articles.

**Application**
- Stream the answer to the page as it is generated, to cut the perceived wait.
- Cache answers to repeated queries, and add request rate limiting for any public deployment.
- Add a language toggle and a right-to-left layout for the interface itself.
- Add automated browser tests for the page.