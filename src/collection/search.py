import os
from dataclasses import dataclass

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


def search_web(query: str, max_results: int) -> list[SearchResult]:
    """
    Runs a web search via Tavily and returns the top results.

    Tavily is used instead of a general-purpose search engine because it returns
    already-ranked, LLM-friendly results (title, url, content snippet) in one call.
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
