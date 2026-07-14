from datetime import datetime, timezone

import requests

from src.collection.models import FetchedPage


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


class FetchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        content_type: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.content_type = content_type


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def fetch_html(url: str, timeout_seconds: int = 15) -> FetchedPage:
    response = requests.get(
        url,
        headers=DEFAULT_HEADERS,
        timeout=timeout_seconds,
        allow_redirects=True,
    )
    content_type = response.headers.get("content-type", "")

    if response.status_code in {401, 403}:
        raise FetchError(
            (
                "site blocked access to this URL. "
                "Use another public URL or a syndicated copy for reproducible scraping."
            ),
            status_code=response.status_code,
            content_type=content_type,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise FetchError(
            f"HTTP request failed: {exc}",
            status_code=response.status_code,
            content_type=content_type,
        ) from exc

    if "html" not in content_type.lower():
        raise FetchError(
            f"URL did not return HTML content: {content_type}",
            status_code=response.status_code,
            content_type=content_type,
        )

    return FetchedPage(
        url=url,
        html=response.text,
        retrieved_at=utc_now_iso(),
        status_code=response.status_code,
        content_type=content_type,
    )
