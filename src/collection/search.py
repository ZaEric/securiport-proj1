import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # type: ignore[assignment]


class SearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str


def build_search_query(target_name: str) -> str:
    return f'"{target_name}" news'

def search_web(*, search_provider: str, query: str, max_results: int, run_dir: Path) -> list[SearchResult]:
    if search_provider == "tavily":
        return search_tavily(query=query, max_results=max_results, run_dir=run_dir)

    raise SearchError(f"Unsupported search_provider: {search_provider}")

def search_tavily(*, query: str, max_results: int, run_dir: Path) -> list[SearchResult]:
    """
    Runs a web search via Tavily and returns the top results.

    Tavily is used instead of a general-purpose search engine because it returns
    already-ranked, LLM-friendly results (title, url, content snippet) in one call.

    Also save Tavily json response for debug help.
    """
    if TavilyClient is None:
        raise SearchError("tavily-python is not installed. Add it to environment.yml.")

    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise SearchError("TAVILY_API_KEY is not set. Add it to your .env file.")

    client = TavilyClient(api_key=api_key)

    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )
    except Exception as exc:
        raise SearchError(f"Tavily search request failed: {exc}") from exc

    save_search_response_json(
        run_dir=run_dir,
        search_provider="Tavily",
        search_query=query,
        response=response,
    )

    results = response.get("results", []) if isinstance(response, dict) else []

    return [
        SearchResult(
            url=item["url"],
            title=item.get("title", ""),
            snippet=item.get("content", ""),
        )
        for item in results
        if item.get("url")
    ]

def save_search_response_json(*, run_dir: Path, search_provider: str, search_query: str, response: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    filename = build_search_response_filename(
        search_provider=search_provider,
        search_query=search_query,
    )

    output_path = run_dir / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(response, file, indent=2, ensure_ascii=False)


def build_search_response_filename(*, search_provider: str, search_query: str) -> str:
    cleaned_query = search_query.replace('"', "")
    cleaned_query = re.sub(r"[<>:/\\|?*]", "", cleaned_query)
    cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

    cleaned_provider = search_provider.strip()

    return f"{cleaned_provider}_{cleaned_query}.json"