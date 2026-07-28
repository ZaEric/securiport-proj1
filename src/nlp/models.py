import os
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


SentimentLabel = Literal["positive", "neutral", "negative"]


def get_default_evidence_model_name() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_EVIDENCE_MODEL", "gpt-oss:20b")


def get_default_report_model_name() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_REPORT_MODEL", "gpt-oss:20b")


@dataclass
class EvidenceExtractionConfig:
    model_name: str = field(default_factory=get_default_evidence_model_name)
    temperature: float = 0.0
    seed: int | None = 42
    evidence_extraction_method: str = "llm_quote_extraction"


@dataclass
class ReportGenerationConfig:
    model_name: str = field(default_factory=get_default_report_model_name)
    temperature: float = 0.0
    seed: int | None = 42


def build_llm_metadata(config: EvidenceExtractionConfig | ReportGenerationConfig) -> dict[str, Any]:
    return {
        "model_name": config.model_name,
        "temperature": config.temperature,
        "seed": config.seed,
    }


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