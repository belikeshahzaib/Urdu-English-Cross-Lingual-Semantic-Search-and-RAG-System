import argparse
from collections import deque
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "corpus.csv"
USER_AGENT = "UrduEnglishSemanticSearch/1.0 (academic project)"
WIKIPEDIA_LICENSE = "CC BY-SA 4.0"
WIKIPEDIA_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

SEED_TOPICS = {
    "en": [
        "Pakistan",
        "History of Pakistan",
        "Geography of Pakistan",
        "Culture of Pakistan",
        "Languages of Pakistan",
        "Economy of Pakistan",
        "Politics of Pakistan",
        "Education in Pakistan",
        "Health in Pakistan",
        "Climate of Pakistan",
        "Karachi",
        "Lahore",
        "Islamabad",
        "Indus River",
        "Indus Valley Civilisation",
        "Mughal Empire",
        "Pakistan Movement",
        "Muhammad Ali Jinnah",
        "Benazir Bhutto",
        "Khyber Pakhtunkhwa",
        "Punjab, Pakistan",
        "Sindh",
        "Balochistan, Pakistan",
    ],
    "ur": [
        "پاکستان",
        "تاریخ پاکستان",
        "پاکستان کا جغرافیہ",
        "پاکستان کی ثقافت",
        "پاکستان کی زبانیں",
        "پاکستان کی معیشت",
        "پاکستان کی سیاست",
        "پاکستان میں تعلیم",
        "پاکستان میں صحت",
        "پاکستان کی آب و ہوا",
        "کراچی",
        "لاہور",
        "اسلام آباد",
        "دریائے سندھ",
        "وادی سندھ کی تہذیب",
        "مغلیہ سلطنت",
        "تحریک پاکستان",
        "محمد علی جناح",
        "بے نظیر بھٹو",
        "خیبر پختونخوا",
        "پنجاب، پاکستان",
        "سندھ",
        "بلوچستان، پاکستان",
    ],
}


def fetch_wikipedia_articles(
    topics: list[str],
    language: str,
    max_pages: int = 5000,
) -> pd.DataFrame:
    if language not in {"en", "ur"}:
        raise ValueError("language must be 'en' or 'ur'")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")

    try:
        import wikipediaapi
    except ImportError as error:
        raise RuntimeError(
            "Wikipedia collection requires the 'wikipedia-api' dependency."
        ) from error

    wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language=language)
    queue = deque()
    scheduled: set[str] = set()
    for topic in topics:
        normalized_topic = topic.strip().casefold()
        if normalized_topic and normalized_topic not in scheduled:
            scheduled.add(normalized_topic)
            queue.append(topic)
    articles: list[dict[str, str]] = []

    while queue and len(articles) < max_pages:
        title = queue.popleft()
        page = wiki.page(title)
        if not page.exists() or page.namespace != wikipediaapi.Namespace.MAIN:
            continue

        text = page.text.strip()
        if not text:
            continue

        articles.append(
            {
                "title": page.title,
                "text": text,
                "url": page.fullurl,
                "lang": language,
                "license": WIKIPEDIA_LICENSE,
                "license_url": WIKIPEDIA_LICENSE_URL,
            }
        )

        for linked_page in page.links.values():
            if linked_page.namespace == wikipediaapi.Namespace.MAIN:
                linked_title = linked_page.title
                normalized_link = linked_title.casefold()
                if normalized_link not in scheduled:
                    scheduled.add(normalized_link)
                    queue.append(linked_title)

    return pd.DataFrame(
        articles,
        columns=["title", "text", "url", "lang", "license", "license_url"],
    )


def collect_corpus(
    output_path: Path = DEFAULT_OUTPUT,
    max_pages_per_language: int = 5000,
) -> pd.DataFrame:
    frames = [
        fetch_wikipedia_articles(topics, language, max_pages_per_language)
        for language, topics in SEED_TOPICS.items()
    ]
    corpus = pd.concat(frames, ignore_index=True)
    corpus = corpus.drop_duplicates(subset=["lang", "title"], keep="first")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output_path, index=False, encoding="utf-8")
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect linked English and Urdu Wikipedia articles."
    )
    parser.add_argument(
        "--max-pages-per-language",
        type=int,
        default=5000,
        help="Maximum number of articles to collect for each language.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    corpus = collect_corpus(args.output, args.max_pages_per_language)
    print(f"Saved {len(corpus)} articles to {args.output}")


if __name__ == "__main__":
    main()
