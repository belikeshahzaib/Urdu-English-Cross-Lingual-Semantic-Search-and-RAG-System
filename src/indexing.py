import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = ROOT / "data" / "processed" / "chunks.csv"
MODEL_NAMES = {
    "e5": "intfloat/multilingual-e5-small",
    "bge-m3": "BAAI/bge-m3",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def encode_passages(model: object, texts: list[str], model_name: str) -> np.ndarray:
    if "e5" in model_name.casefold():
        texts = [f"passage: {text}" for text in texts]
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def build_indexes(
    chunks_path: Path = DEFAULT_CHUNKS,
    index_dir: Path | None = None,
    model_name: str = MODEL_NAMES["e5"],
) -> tuple[Path, Path]:
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunk file not found at {chunks_path}. Run src.preprocessing first."
        )

    frame = pd.read_csv(chunks_path, keep_default_na=False)
    if "text" not in frame or frame.empty:
        raise ValueError(f"No chunk text found in {chunks_path}")

    try:
        import faiss
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Index building requires faiss-cpu, rank-bm25, and sentence-transformers."
        ) from error

    texts = frame["text"].astype(str).tolist()
    model = SentenceTransformer(model_name)
    embeddings = encode_passages(model, texts, model_name)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("The embedding model returned an invalid embedding matrix.")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    bm25 = BM25Okapi([tokenize(text) for text in texts])

    if index_dir is None:
        model_key = next(
            (key for key, name in MODEL_NAMES.items() if name == model_name),
            "custom",
        )
        index_dir = ROOT / "indexes" / model_key
    index_dir.mkdir(parents=True, exist_ok=True)
    dense_path = index_dir / "vector_index.faiss"
    sparse_path = index_dir / "bm25_index.pkl"
    faiss.write_index(index, str(dense_path))
    with sparse_path.open("wb") as file:
        pickle.dump(bm25, file, protocol=pickle.HIGHEST_PROTOCOL)
    return dense_path, sparse_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS and BM25 indexes.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index-dir", type=Path)
    parser.add_argument(
        "--model", choices=sorted(MODEL_NAMES), default="e5",
        help="Embedding model used to build the dense index.",
    )
    args = parser.parse_args()
    model_name = MODEL_NAMES[args.model]
    dense_path, sparse_path = build_indexes(
        args.chunks, args.index_dir, model_name
    )
    print(f"Saved dense index to {dense_path}")
    print(f"Saved BM25 index to {sparse_path}")


if __name__ == "__main__":
    main()
