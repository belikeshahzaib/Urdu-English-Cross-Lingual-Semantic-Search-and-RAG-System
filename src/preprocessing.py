import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "raw" / "corpus.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "chunks.csv"
DEFAULT_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
URDU_TRANSLATION = str.maketrans(
    {
        "\u064a": "\u06cc",
        "\u0649": "\u06cc",
        "\u0643": "\u06a9",
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0671": "\u0627",
    }
)
DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
MARKUP = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


def normalize_urdu_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = normalized.replace("\u0640", "")
    normalized = DIACRITICS.sub("", normalized)
    return normalized.translate(URDU_TRANSLATION)


def clean_text(text: str) -> str:
    text = normalize_urdu_text(text)
    text = MARKUP.sub(" ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s\u0600-\u06ff]", " ", text, flags=re.UNICODE)
    return WHITESPACE.sub(" ", text).strip()


def clean_urdu_text(text: str) -> str:
    return clean_text(text)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size - 1")

    words = text.split()
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


def create_chunks(
    corpus: pd.DataFrame,
    chunk_size: int = 200,
    overlap: int = 50,
) -> pd.DataFrame:
    required = {"title", "text", "url", "lang"}
    missing = required.difference(corpus.columns)
    if missing:
        missing_columns = ", ".join(sorted(missing))
        raise ValueError(f"Corpus is missing required columns: {missing_columns}")

    records: list[dict[str, str]] = []
    for row in corpus.itertuples(index=False):
        cleaned_text = clean_text(row.text)
        for chunk_number, chunk in enumerate(
            chunk_text(cleaned_text, chunk_size, overlap)
        ):
            records.append(
                {
                    "doc_title": str(row.title),
                    "chunk_id": f"{row.lang}:{row.title}:{chunk_number}",
                    "text": chunk,
                    "url": str(row.url),
                    "lang": str(row.lang),
                    "license": str(getattr(row, "license", "CC BY-SA 4.0")),
                    "license_url": str(
                        getattr(row, "license_url", DEFAULT_LICENSE_URL)
                    ),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            "doc_title",
            "chunk_id",
            "text",
            "url",
            "lang",
            "license",
            "license_url",
        ],
    )


def preprocess_corpus(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    chunk_size: int = 200,
    overlap: int = 50,
) -> pd.DataFrame:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Corpus file not found at {input_path}. Run src.data_collection first."
        )
    corpus = pd.read_csv(input_path, keep_default_na=False)
    chunks = create_chunks(corpus, chunk_size, overlap)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_csv(output_path, index=False, encoding="utf-8")
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and chunk the bilingual corpus."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    chunks = preprocess_corpus(
        args.input, args.output, args.chunk_size, args.overlap
    )
    print(f"Saved {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()
