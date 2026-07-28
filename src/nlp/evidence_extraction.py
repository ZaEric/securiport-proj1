import json
import re
from typing import Any

from src.nlp.llm_client import LLMClient
from src.nlp.models import EvidenceExtractionConfig, ExtractedQuote
from src.nlp.prompts import build_evidence_extraction_prompt
from src.nlp.quote_verification import quote_exists_in_text


VALID_SENTIMENTS = {"positive", "neutral", "negative"}


def extract_evidence_from_chunk_llm(
    target_name: str,
    chunk_text: str,
    llm_client: LLMClient,
    config: EvidenceExtractionConfig | None = None,
) -> list[ExtractedQuote]:
    config = config or EvidenceExtractionConfig()

    prompt = build_evidence_extraction_prompt(
        target_name=target_name,
        chunk_text=chunk_text,
    )

    raw_response = llm_client.generate_json(
        model=config.model_name,
        prompt=prompt,
        temperature=config.temperature,
        seed=config.seed,
    )

    parsed = parse_llm_json_response(raw_response)
    raw_items = parsed.get("evidence", [])

    if not isinstance(raw_items, list):
        return []

    extracted_quotes: list[ExtractedQuote] = []

    for item in raw_items:
        extracted = parse_evidence_item(item)

        if extracted is None:
            continue

        if not quote_exists_in_text(extracted.quote, chunk_text):
            continue

        extracted_quotes.append(extracted)

    return extracted_quotes


def parse_evidence_item(item: Any) -> ExtractedQuote | None:
    if not isinstance(item, dict):
        return None

    quote = item.get("quote")
    sentiment = item.get("sentiment")

    if not isinstance(quote, str):
        return None

    if not isinstance(sentiment, str):
        return None

    sentiment = sentiment.strip().lower()

    if sentiment not in VALID_SENTIMENTS:
        return None

    quote = quote.strip()

    if not quote:
        return None

    return ExtractedQuote(
        quote=quote,
        sentiment=sentiment,  # type: ignore[arg-type]
    )


def parse_llm_json_response(raw_response: str) -> dict[str, Any]:
    """
    Parses JSON returned by the LLM.

    Ollama JSON mode should usually return clean JSON, but this also handles
    common cases like accidental markdown code fences.
    """
    cleaned = strip_code_fences(raw_response).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = json.loads(extract_first_json_object(cleaned))

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")

    return parsed


def strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {text}")

    return text[start : end + 1]