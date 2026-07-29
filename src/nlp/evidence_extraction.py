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
) -> tuple[list[ExtractedQuote], list[dict[str, Any]]]:
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
        return [], [
            {
                "reason": "raw_evidence_not_list",
                "raw_evidence_type": type(raw_items).__name__,
                "raw_evidence_preview": truncate_for_issue(raw_items),
            }
        ]

    extracted_quotes: list[ExtractedQuote] = []
    extraction_issues: list[dict[str, Any]] = []

    for item in raw_items:
        extracted, issue = parse_evidence_item_with_issue(item)

        if issue is not None:
            extraction_issues.append(issue)
            continue

        if extracted is None:
            extraction_issues.append(
                {
                    "reason": "unknown_parse_failure",
                    "raw_item_preview": truncate_for_issue(item),
                }
            )
            continue

        if not quote_exists_in_text(extracted.quote, chunk_text):
            extraction_issues.append(
                {
                    "reason": "quote_not_found_in_chunk",
                    "quote": extracted.quote,
                    "sentiment": extracted.sentiment,
                }
            )
            continue

        extracted_quotes.append(extracted)

    return extracted_quotes, extraction_issues


def parse_evidence_item_with_issue(item: Any) -> tuple[ExtractedQuote | None, dict[str, Any] | None]:
    if not isinstance(item, dict):
        return None, {
            "reason": "evidence_item_not_object",
            "raw_item_type": type(item).__name__,
            "raw_item_preview": truncate_for_issue(item),
        }

    quote = item.get("quote")
    sentiment = item.get("sentiment")

    if not isinstance(quote, str):
        return None, {
            "reason": "missing_or_invalid_quote",
            "quote_type": type(quote).__name__,
            "sentiment": sentiment,
            "raw_item_preview": truncate_for_issue(item),
        }

    if not isinstance(sentiment, str):
        return None, {
            "reason": "missing_or_invalid_sentiment",
            "quote": quote,
            "sentiment_type": type(sentiment).__name__,
            "raw_item_preview": truncate_for_issue(item),
        }

    normalized_sentiment = sentiment.strip().lower()

    if normalized_sentiment not in VALID_SENTIMENTS:
        return None, {
            "reason": "invalid_sentiment_label",
            "quote": quote,
            "sentiment": normalized_sentiment,
            "valid_sentiments": sorted(VALID_SENTIMENTS),
        }

    normalized_quote = quote.strip()

    if not normalized_quote:
        return None, {
            "reason": "empty_quote",
            "sentiment": normalized_sentiment,
            "raw_item_preview": truncate_for_issue(item),
        }

    return ExtractedQuote(
        quote=normalized_quote,
        sentiment=normalized_sentiment,  # type: ignore[arg-type]
    )

def parse_evidence_item(item: Any) -> ExtractedQuote | None:
    """
    Backward-compatible parser for callers that only want the parsed quote.
    """
    extracted, _issue = parse_evidence_item_with_issue(item)
    return extracted

def truncate_for_issue(value: Any, max_chars: int = 500) -> str:
    """
    Keeps evidence_extraction_issues readable and prevents very large raw items
    from making evidence.json noisy.
    """
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "...[truncated]"

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