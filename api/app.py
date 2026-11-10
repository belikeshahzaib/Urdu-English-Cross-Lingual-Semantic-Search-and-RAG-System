from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from src.retrieval import Retriever


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"
INDEX_DIR = ROOT / "indexes" / "e5"
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.csv"
MODEL_NAME = "intfloat/multilingual-e5-small"

app = FastAPI(title="Urdu-English Semantic Search", version="1.0.0")


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
