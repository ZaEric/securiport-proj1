from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceDocument:
    source_id: str
    url: str
    title: str
    article_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    chunk_id: str
    source_id: str
    text: str
    word_count: int