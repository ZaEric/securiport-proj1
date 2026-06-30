from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_synthetic_input_json(
    *,
    run_id: str,
    target_name: str,
    synthetic_case_file: str,
    expected_overall_sentiment: str | None = None,
    difficulty_tags: list[str] | None = None,
    expected_failure_modes: list[str] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "target_name": target_name,
        "input_mode": "synthetic",
        "synthetic_case_file": synthetic_case_file,
        "expected_overall_sentiment": expected_overall_sentiment,
        "difficulty_tags": difficulty_tags or [],
        "expected_failure_modes": expected_failure_modes or [],
        "notes": notes or "Synthetic case used for controlled pipeline testing.",
    }


def build_curated_urls_input_json(
    *,
    run_id: str,
    target_name: str,
    curated_urls_file: str,
    requested_urls: list[str],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "target_name": target_name,
        "input_mode": "curated_urls",
        "curated_urls_file": curated_urls_file,
        "requested_urls": requested_urls,
    }


def build_search_api_input_json(
    *,
    run_id: str,
    target_name: str,
    search_provider: str,
    search_query: str,
    max_urls: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "target_name": target_name,
        "input_mode": "search_api",
        "search_provider": search_provider,
        "search_query": search_query,
        "max_urls": max_urls,
    }


def build_sources_json_from_synthetic(
    *,
    run_id: str,
    target_name: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Converts the synthetic dataset's sources into the common sources.json artifact.

    Keeps expected_source_sentiment if present, since synthetic data may include it
    for evaluation/debugging.
    """
    return {
        "run_id": run_id,
        "target_name": target_name,
        "sources": sources,
        "metadata": {
            "source_mode": "synthetic",
            "source_count": len(sources),
        },
    }