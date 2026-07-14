from bs4 import BeautifulSoup

try:
    import trafilatura
except ImportError:
    trafilatura = None

from src.collection.models import ExtractedPage, FetchedPage


def extract_page_text(page: FetchedPage) -> ExtractedPage:
    title = extract_title(page.html)
    article_text = extract_with_trafilatura(page.html, page.url)

    if not article_text:
        article_text = extract_with_beautifulsoup(page.html)

    return ExtractedPage(
        url=page.url,
        title=title,
        article_text=article_text.strip(),
        retrieved_at=page.retrieved_at,
        status_code=page.status_code,
        content_type=page.content_type,
    )


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.title and soup.title.string:
        return soup.title.string.strip()

    heading = soup.find("h1")

    if heading:
        return heading.get_text(" ", strip=True)

    return ""


def extract_with_trafilatura(html: str, url: str) -> str:
    if trafilatura is None:
        return ""

    extracted = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
    )
    return extracted or ""


def extract_with_beautifulsoup(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # remove common non-content elements.
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
