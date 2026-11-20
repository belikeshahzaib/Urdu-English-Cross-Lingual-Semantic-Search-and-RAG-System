from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.generation import (
    ConfigurationError,
    ProviderError,
    generate_answer,
    load_env_file,
)
from src.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
INDEX_DIR = ROOT / "indexes" / "e5"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.csv"
MODEL_NAME = "intfloat/multilingual-e5-small"

# Provider settings (LLM_PROVIDER, LLM_MODEL, *_API_KEY) come from a local .env.
load_env_file(ROOT / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the retriever at startup so the first request is fast."""
    try:
        get_retriever()
    except (FileNotFoundError, RuntimeError, ValueError):
        pass  # /health and the endpoints report problems later
    yield


app = FastAPI(
    title="Urdu-English Semantic Search", version="1.0.0", lifespan=lifespan
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(
        CHUNKS_PATH,
        INDEX_DIR / "vector_index.faiss",
        INDEX_DIR / "bm25_index.pkl",
        MODEL_NAME,
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(FRONTEND)


@app.get("/health")
def health() -> dict[str, str]:
    required = [
        CHUNKS_PATH,
        INDEX_DIR / "vector_index.faiss",
        INDEX_DIR / "bm25_index.pkl",
    ]
    if not all(path.is_file() for path in required):
        return {"status": "not_ready"}
    return {"status": "ready"}


@app.post("/search")
def search(request: QueryRequest) -> dict[str, list[dict[str, object]]]:
    try:
        results = get_retriever().retrieve(request.query, k=request.k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"results": results}


@app.post("/answer")
def answer(request: QueryRequest) -> dict[str, object]:
    """Retrieve passages, then generate a cited answer from them only."""
    try:
        passages = get_retriever().retrieve(request.query, k=request.k)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        result = generate_answer(request.query, passages)
    except ConfigurationError as error:
        raise HTTPException(
            status_code=503, detail=f"Answer generation is not configured: {error}"
        ) from error
    except ProviderError as error:
        raise HTTPException(
            status_code=502, detail=f"The language model provider failed: {error}"
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return result.to_dict()