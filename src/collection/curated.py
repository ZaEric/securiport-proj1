from pathlib import Path
from typing import Any

from src.api.io import read_json
from src.collection.models import CuratedPersonEntry, CuratedUrl


def load_curated_person_entry(
    curated_urls_file: str,
    target_name: str,
) -> CuratedPersonEntry:
    path = Path(curated_urls_file)
    data = read_json(path)

    if not isinstance(data, list):
        raise ValueError("curated URLs file must contain a list of person records.")

    for item in data:
        entry = parse_curated_person_entry(item)

        if entry.person.lower() == target_name.strip().lower():
            return entry

    raise ValueError(f"target_name not found in curated URLs file: {target_name}")


def parse_curated_person_entry(item: Any) -> CuratedPersonEntry:
    if not isinstance(item, dict):
        raise ValueError("each curated person record must be an object.")

    person = item.get("person")

    if not isinstance(person, str) or not person.strip():
        raise ValueError("curated person record missing person.")

    raw_urls = item.get("urls")

    if not isinstance(raw_urls, list):
        raise ValueError(f"curated person record missing urls list: {person}")

    urls = [parse_curated_url(raw_url) for raw_url in raw_urls]
    expected_sentiment = item.get("expected_sentiment")

    return CuratedPersonEntry(
        person=person.strip(),
        expected_sentiment=expected_sentiment if isinstance(expected_sentiment, str) else None,
        urls=urls,
    )


def parse_curated_url(item: Any) -> CuratedUrl:
    if not isinstance(item, dict):
        raise ValueError("each curated URL record must be an object.")

    url = item.get("url")

    if not isinstance(url, str):
        raise ValueError("curated URL record missing url.")

    source_type = item.get("source_type", "")
    notes = item.get("notes", "")

    return CuratedUrl(
        url=url,
        source_type=source_type if isinstance(source_type, str) else "",
        notes=notes if isinstance(notes, str) else "",
    )

