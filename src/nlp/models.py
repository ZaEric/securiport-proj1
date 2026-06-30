import os
from dataclasses import dataclass
from typing import Literal


SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass
class EvidenceExtractionConfig:
    model_name: str = os.getenv("OLLAMA_MODEL", "gpt-oss:120b")
    temperature: float = 0.0
    evidence_extraction_method: str = "llm_quote_extraction"


@dataclass
class ExtractedQuote:
    quote: str
    sentiment: SentimentLabel


@dataclass
class EvidenceSnippet:
    evidence_id: str
    source_id: str
    chunk_id: str
    quote: str
    sentiment: SentimentLabel
    confidence: None = None