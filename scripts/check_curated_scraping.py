import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.io import write_json
from src.api.run_artifacts import build_curated_urls_input_json
from src.api.run_ids import generate_run_id
from src.collection.curated import parse_curated_person_entry
from src.collection.pipeline import build_sources_json_from_curated_urls


DEFAULT_CURATED_URLS_FILE = "data/curated/curated_urls.json"


def main() -> None:
    args = parse_args()
    curated_path = Path(args.curated_urls_file)
    records = load_curated_records(curated_path)

    selected_records = filter_records_by_person(records, args.person)

    if not selected_records:
        print("No matching curated records found.")
        return

    summaries: list[dict[str, Any]] = []

    for record in selected_records:
        entry = parse_curated_person_entry(record)
        run_id = generate_run_id(entry.person)

        print(f"\nChecking {entry.person}...")

        sources_json = build_sources_json_from_curated_urls(
            run_id=run_id,
            target_name=entry.person,
            curated_urls_file=args.curated_urls_file,
            max_urls=args.max_urls,
        )

        summary = build_summary(sources_json)
        summaries.append(summary)
        print_summary(summary)

        if args.write_runs:
            write_run_artifacts(
                run_id=run_id,
                curated_urls_file=args.curated_urls_file,
                sources_json=sources_json,
            )

    print_overall_summary(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check scraping and text extraction quality for curated URLs.",
    )
    parser.add_argument(
        "--curated-urls-file",
        default=DEFAULT_CURATED_URLS_FILE,
        help="Path to curated_urls.json.",
    )
    parser.add_argument(
        "--person",
        default=None,
        help="Optional person name to check. If omitted, checks all curated people.",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Optional maximum URLs per person.",
    )
    parser.add_argument(
        "--write-runs",
        action="store_true",
        help="Write input.json and sources.json under data/runs/<run_id>.",
    )
    return parser.parse_args()


def load_curated_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("curated URLs file must contain a list.")

    return data


def filter_records_by_person(
    records: list[dict[str, Any]],
    person: str | None,
) -> list[dict[str, Any]]:
    if person is None:
        return records

    person_key = person.strip().lower()
    return [
        record
        for record in records
        if str(record.get("person", "")).strip().lower() == person_key
    ]


def build_summary(sources_json: dict[str, Any]) -> dict[str, Any]:
    sources = sources_json["sources"]
    metadata = sources_json["metadata"]
    target_name = sources_json["target_name"]

    source_summaries = []

    for source in sources:
        article_text = source.get("article_text", "")
        source_summaries.append(
            {
                "source_id": source.get("source_id", ""),
                "url": source.get("url", ""),
                "text_length": len(article_text),
                "word_count": count_words(article_text),
                "target_name_found": target_name.lower() in article_text.lower(),
            }
        )

    return {
        "run_id": sources_json["run_id"],
        "target_name": target_name,
        "expected_overall_sentiment": metadata.get("expected_overall_sentiment"),
        "requested_url_count": metadata.get("requested_url_count", 0),
        "source_count": metadata.get("source_count", 0),
        "failed_url_count": metadata.get("failed_url_count", 0),
        "blocked_url_count": metadata.get("blocked_url_count", 0),
        "source_summaries": source_summaries,
        "fetch_errors": metadata.get("fetch_errors", []),
    }


def count_words(text: str) -> int:
    return len(text.split())


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "  "
        f"expected={summary['expected_overall_sentiment']} "
        f"requested={summary['requested_url_count']} "
        f"sources={summary['source_count']} "
        f"failed={summary['failed_url_count']} "
        f"blocked={summary['blocked_url_count']}"
    )

    for source in summary["source_summaries"]:
        target_status = "yes" if source["target_name_found"] else "no"
        print(
            "  "
            f"{source['source_id']}: "
            f"words={source['word_count']} "
            f"chars={source['text_length']} "
            f"target_found={target_status}"
        )

    for error in summary["fetch_errors"]:
        print(
            "  "
            f"error status={error.get('status_code')} "
            f"url={error.get('url')} "
            f"message={error.get('error')}"
        )


def write_run_artifacts(
    *,
    run_id: str,
    curated_urls_file: str,
    sources_json: dict[str, Any],
) -> None:
    run_dir = Path("data/runs") / run_id
    input_path = run_dir / "input.json"
    sources_path = run_dir / "sources.json"

    requested_urls = [source["url"] for source in sources_json["sources"]]

    input_json = build_curated_urls_input_json(
        run_id=run_id,
        target_name=sources_json["target_name"],
        curated_urls_file=curated_urls_file,
        requested_urls=requested_urls,
        expected_overall_sentiment=sources_json["metadata"].get(
            "expected_overall_sentiment"
        ),
    )

    write_json(input_path, input_json)
    write_json(sources_path, sources_json)

    print(f"  wrote {input_path.as_posix()}")
    print(f"  wrote {sources_path.as_posix()}")


def print_overall_summary(summaries: list[dict[str, Any]]) -> None:
    total_people = len(summaries)
    total_requested = sum(item["requested_url_count"] for item in summaries)
    total_sources = sum(item["source_count"] for item in summaries)
    total_failed = sum(item["failed_url_count"] for item in summaries)
    total_blocked = sum(item["blocked_url_count"] for item in summaries)

    print("\nOverall")
    print(f"  people={total_people}")
    print(f"  requested_urls={total_requested}")
    print(f"  successful_sources={total_sources}")
    print(f"  failed_urls={total_failed}")
    print(f"  blocked_urls={total_blocked}")


if __name__ == "__main__":
    main()
