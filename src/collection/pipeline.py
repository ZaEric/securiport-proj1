from typing import Any

from src.collection.curated import load_curated_person_entry
from src.collection.extraction import extract_page_text
from src.collection.fetching import FetchError, fetch_html
from src.collection.models import CuratedUrl, ExtractedPage
from src.collection.search import SearchError, search_web
from src.collection.url_filtering import filter_curated_urls

MIN_ARTICLE_WORDS = 50


def fetch_and_extract_sources(
    urls: list[CuratedUrl],
    target_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Shared fetch -> extract -> relevance-check loop used by both the curated-urls
    and search-api source-creation flows, since they only differ in where the
    candidate URLs come from.
    """
    sources: list[dict[str, Any]] = []
    fetch_errors: list[dict[str, Any]] = []

    for item in urls:
        try:
            page = fetch_html(item.url)
            extracted_page = extract_page_text(page)
            if len(extracted_page.article_text.split()) < MIN_ARTICLE_WORDS:
                fetch_errors.append(
                    build_fetch_error_item(
                        curated_url=item,
                        error=f"extracted article text was too short to use: fewer than {MIN_ARTICLE_WORDS} words",
                        status_code=extracted_page.status_code,
                        content_type=extracted_page.content_type,
                    )
                )
                continue
        except FetchError as exc:
            fetch_errors.append(
                build_fetch_error_item(
                    curated_url=item,
                    error=str(exc),
                    status_code=exc.status_code,
                    content_type=exc.content_type,
                )
            )
            continue
        except Exception as exc:
            fetch_errors.append(
                build_fetch_error_item(
                    curated_url=item,
                    error=str(exc),
                )
            )
            continue

        if not is_relevant_to_target(extracted_page, target_name):
            fetch_errors.append(
                build_fetch_error_item(
                    curated_url=item,
                    error=f"extracted text did not mention target_name: {target_name}",
                    status_code=extracted_page.status_code,
                    content_type=extracted_page.content_type,
                )
            )
            continue

        sources.append(
            build_source_item(
                source_index=len(sources),
                curated_url=item,
                extracted_page=extracted_page,
            )
        )

    return sources, fetch_errors


def build_sources_json_from_curated_urls(
    *,
    run_id: str,
    target_name: str,
    curated_urls_file: str,
    max_urls: int | None = None,
) -> dict[str, Any]:
    entry = load_curated_person_entry(
        curated_urls_file=curated_urls_file,
        target_name=target_name,
    )
    curated_urls = filter_curated_urls(entry.urls, max_urls=max_urls)

    sources, fetch_errors = fetch_and_extract_sources(curated_urls, target_name)

    return {
        "run_id": run_id,
        "target_name": target_name,
        "sources": sources,
        "metadata": {
            "source_mode": "curated_urls",
            "source_count": len(sources),
            "requested_url_count": len(curated_urls),
            "failed_url_count": len(fetch_errors),
            "blocked_url_count": count_blocked_urls(fetch_errors),
            "expected_overall_sentiment": entry.expected_sentiment,
            "fetch_errors": fetch_errors,
        },
    }


def build_sources_json_from_search_api(
    *,
    run_id: str,
    target_name: str,
    search_provider: str,
    search_query: str,
    max_urls: int,
) -> dict[str, Any]:
    try:
        search_results = search_web(search_query, max_results=max_urls)
    except SearchError as exc:
        return {
            "run_id": run_id,
            "target_name": target_name,
            "sources": [],
            "metadata": {
                "source_mode": "search_api",
                "search_provider": search_provider,
                "search_query": search_query,
                "source_count": 0,
                "requested_url_count": 0,
                "failed_url_count": 0,
                "blocked_url_count": 0,
                "fetch_errors": [{"url": None, "source_type": "", "notes": "", "status_code": None, "content_type": "", "error": str(exc)}],
            },
        }

    candidate_urls = [
        CuratedUrl(url=item.url, source_type="search_result", notes=item.title)
        for item in search_results
    ]
    filtered_urls = filter_curated_urls(candidate_urls, max_urls=max_urls)

    sources, fetch_errors = fetch_and_extract_sources(filtered_urls, target_name)

    return {
        "run_id": run_id,
        "target_name": target_name,
        "sources": sources,
        "metadata": {
            "source_mode": "search_api",
            "search_provider": search_provider,
            "search_query": search_query,
            "source_count": len(sources),
            "requested_url_count": len(filtered_urls),
            "failed_url_count": len(fetch_errors),
            "blocked_url_count": count_blocked_urls(fetch_errors),
            "fetch_errors": fetch_errors,
        },
    }

# changed relevance checking to be more generous, check full name plus last name
def is_relevant_to_target(page: ExtractedPage, target_name: str) -> bool:
    normalized_name = target_name.strip().lower()
    name_parts = normalized_name.split()
    last_name = name_parts[-1] if name_parts else ""

    combined_text = f"{page.title}\n{page.article_text}".lower()

    return normalized_name in combined_text or (
        len(last_name) >= 3 and last_name in combined_text
    )


def build_source_item(
    source_index: int,
    curated_url: CuratedUrl,
    extracted_page: ExtractedPage,
) -> dict[str, Any]:
    return {
        "source_id": f"src_{source_index + 1:03d}",
        "url": extracted_page.url,
        "title": extracted_page.title,
        "retrieved_at": extracted_page.retrieved_at,
        "article_text": extracted_page.article_text,
        "source_type": curated_url.source_type,
        "notes": curated_url.notes,
        "status_code": extracted_page.status_code,
        "content_type": extracted_page.content_type,
    }


def build_fetch_error_item(
    *,
    curated_url: CuratedUrl,
    error: str,
    status_code: int | None = None,
    content_type: str = "",
) -> dict[str, Any]:
    return {
        "url": curated_url.url,
        "source_type": curated_url.source_type,
        "notes": curated_url.notes,
        "status_code": status_code,
        "content_type": content_type,
        "error": error,
    }


def count_blocked_urls(fetch_errors: list[dict[str, Any]]) -> int:
    blocked_status_codes = {401, 403}
    return sum(
        1
        for item in fetch_errors
        if item.get("status_code") in blocked_status_codes
    )
