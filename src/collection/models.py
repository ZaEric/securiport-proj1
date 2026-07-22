from dataclasses import dataclass


@dataclass
class CuratedUrl:
    url: str
    source_type: str = ""
    notes: str = ""


@dataclass
class CuratedPersonEntry:
    person: str
    expected_sentiment: str | None
    urls: list[CuratedUrl]


@dataclass
class FetchedPage:
    url: str
    html: str
    retrieved_at: str
    status_code: int
    content_type: str


@dataclass
class ExtractedPage:
    url: str
    title: str
    article_text: str
    retrieved_at: str
    status_code: int
    content_type: str

