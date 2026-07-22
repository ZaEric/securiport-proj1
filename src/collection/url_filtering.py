from urllib.parse import urlparse

from src.collection.models import CuratedUrl


BLOCKED_EXTENSIONS = {
    ".avi",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4v",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".svg",
    ".webp",
    ".zip",
}


def filter_curated_urls(
    urls: list[CuratedUrl],
    max_urls: int | None = None,
) -> list[CuratedUrl]:
    filtered_urls: list[CuratedUrl] = []
    seen_urls: set[str] = set()

    for item in urls:
        normalized_url = item.url.strip()

        if not normalized_url:
            continue

        if normalized_url in seen_urls:
            continue

        if has_blocked_extension(normalized_url):
            continue

        seen_urls.add(normalized_url)
        filtered_urls.append(
            CuratedUrl(
                url=normalized_url,
                source_type=item.source_type,
                notes=item.notes,
            )
        )

        if max_urls is not None and len(filtered_urls) >= max_urls:
            break

    return filtered_urls


def has_blocked_extension(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(extension) for extension in BLOCKED_EXTENSIONS)
