import json
import re
from typing import Any

from src.nlp.llm_client import LLMClient
from src.nlp.models import EvidenceExtractionConfig
from src.nlp.prompts import build_final_report_prompt


VALID_SENTIMENTS = {"positive", "neutral", "negative"}
REQUIRED_REPORT_FIELDS = {
    "overall_sentiment",
    "one_line_summary",
    "extended_summary",
    "key_findings",
    "justification",
    "evidence_examples",
    "sources",
    "limitations",
}


def generate_final_report_llm(
    sources_json: dict[str, Any],
    evidence_json: dict[str, Any],
    aggregation_json: dict[str, Any],
    llm_client: LLMClient,
    config: EvidenceExtractionConfig | None = None,
    max_evidence_examples: int = 7,
) -> dict[str, Any]:
    """
    Generates final_report.json using sources, evidence, and aggregation.

    The LLM writes the narrative report, but selected evidence examples are
    validated against evidence.json after generation.
    """
    config = config or EvidenceExtractionConfig()

    run_id = evidence_json["run_id"]
    target_name = evidence_json["target_name"]

    sources_summary = build_sources_summary(sources_json=sources_json)
    aggregation_summary = build_aggregation_summary(
        evidence_json=evidence_json,
        aggregation_json=aggregation_json,
    )

    prompt = build_final_report_prompt(
        target_name=target_name,
        sources_summary=sources_summary,
        evidence_json=evidence_json,
        aggregation_summary=aggregation_summary,
        max_evidence_examples=max_evidence_examples,
    )

    raw_response = llm_client.generate_json(
        model=config.model_name,
        prompt=prompt,
        temperature=config.temperature,
    )

    parsed_report = parse_llm_json_response(raw_response)

    final_report = normalize_final_report(
        run_id=run_id,
        target_name=target_name,
        parsed_report=parsed_report,
        sources_summary=sources_summary,
        evidence_json=evidence_json,
        aggregation_json=aggregation_json,
        max_evidence_examples=max_evidence_examples,
    )

    return final_report


def build_sources_summary(
    sources_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Builds source metadata for the final report prompt.

    Excludes article_text and excludes aggregation baseline sentiment.
    """
    sources_summary: list[dict[str, Any]] = []

    for source in sources_json.get("sources", []):
        sources_summary.append(
            {
                "source_id": source.get("source_id", ""),
                "url": source.get("url", ""),
                "title": source.get("title", ""),
            }
        )

    return sources_summary

def build_aggregation_summary(
    evidence_json: dict[str, Any],
    aggregation_json: dict[str, Any],
) -> dict[str, Any]:
    """
    Builds compact evidence statistics for the final report prompt.

    Excludes baseline_sentiment, thresholds, source_results, and implementation details.
    """
    overall_result = aggregation_json.get("overall_result", {})
    sources = aggregation_json.get("source_results", [])

    return {
        "num_evidence": evidence_json.get("num_evidence", len(evidence_json.get("evidence", []))),
        "source_count": len(sources),
        "sentiment_counts": overall_result.get("sentiment_counts", {}),
        "mixed_evidence": overall_result.get("mixed_evidence"),
        "insufficient_evidence": overall_result.get("insufficient_evidence"),
    }


def normalize_final_report(
    run_id: str,
    target_name: str,
    parsed_report: dict[str, Any],
    sources_summary: list[dict[str, Any]],
    evidence_json: dict[str, Any],
    aggregation_json: dict[str, Any],
    max_evidence_examples: int,
) -> dict[str, Any]:
    """
    Normalizes and validates the LLM report output before writing final_report.json.

    Does not silently invent fallback sentiment. If required fields are missing
    or invalid, the issue is recorded in report_generation_issues.
    """
    report_generation_issues = validate_report_output(parsed_report)

    overall_sentiment = normalize_required_overall_sentiment(
        parsed_report.get("overall_sentiment"),
        report_generation_issues,
    )

    evidence_examples = validate_evidence_examples(
        raw_examples=parsed_report.get("evidence_examples", []),
        evidence_json=evidence_json,
        max_evidence_examples=max_evidence_examples,
        report_generation_issues=report_generation_issues,
    )

    return {
        "run_id": run_id,
        "target_name": target_name,
        "overall_sentiment": overall_sentiment,
        "overall_confidence": None,
        "aggregation_baseline_sentiment": aggregation_json["overall_result"].get("baseline_sentiment"),
        "one_line_summary": normalize_string(parsed_report.get("one_line_summary")),
        "extended_summary": normalize_string(parsed_report.get("extended_summary")),
        "key_findings": normalize_string_list(parsed_report.get("key_findings")),
        "justification": normalize_string(parsed_report.get("justification")),
        "evidence_examples": evidence_examples,
        "sources": normalize_sources(parsed_report.get("sources"), sources_summary),
        "limitations": normalize_string_list(parsed_report.get("limitations")),
        "report_generation_issues": report_generation_issues,
    }


def normalize_required_overall_sentiment(
    value: Any,
    report_generation_issues: list[str],
) -> str | None:
    """
    Validates required overall_sentiment.

    No silent fallback is used. If the LLM fails to return a valid label, this
    records an issue and returns None.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in VALID_SENTIMENTS:
            return normalized

    report_generation_issues.append(
        "Missing or invalid overall_sentiment. Expected one of: positive, neutral, negative."
    )
    return None


def validate_report_output(parsed_report: dict[str, Any]) -> list[str]:
    """
    Records missing required fields from the LLM-generated report.
    """
    issues: list[str] = []

    for field in sorted(REQUIRED_REPORT_FIELDS):
        if field not in parsed_report:
            issues.append(f"Missing required report field: {field}")

    return issues


def validate_evidence_examples(
    raw_examples: Any,
    evidence_json: dict[str, Any],
    max_evidence_examples: int,
    report_generation_issues: list[str],
) -> list[dict[str, Any]]:
    """
    Keeps only evidence examples that match real evidence quotes.

    Final report examples do not expose evidence_id because it is internal.
    Validation can match by evidence_id if provided, or by exact quote.
    """
    if not isinstance(raw_examples, list):
        report_generation_issues.append("evidence_examples was missing or not a list.")
        return []

    evidence_by_id = {
        item["evidence_id"]: item
        for item in evidence_json.get("evidence", [])
    }
    evidence_by_quote = {
        item["quote"]: item
        for item in evidence_json.get("evidence", [])
    }

    validated_examples: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    invalid_example_count = 0

    for example in raw_examples:
        if not isinstance(example, dict):
            invalid_example_count += 1
            continue

        source_evidence = find_matching_evidence_example(
            example=example,
            evidence_by_id=evidence_by_id,
            evidence_by_quote=evidence_by_quote,
        )

        if source_evidence is None:
            invalid_example_count += 1
            continue

        quote = source_evidence["quote"]

        if quote in seen_quotes:
            continue

        validated_examples.append(
            {
                "source_id": source_evidence["source_id"],
                "quote": source_evidence["quote"],
                "sentiment": source_evidence["sentiment"],
                "why_selected": normalize_string(example.get("why_selected")),
            }
        )
        seen_quotes.add(quote)

        if len(validated_examples) >= max_evidence_examples:
            break

    if invalid_example_count > 0:
        report_generation_issues.append(
            f"Rejected {invalid_example_count} invalid or unverified evidence example(s)."
        )

    if not validated_examples:
        report_generation_issues.append(
            "No valid evidence examples were selected for the final report."
        )

    return validated_examples


def find_matching_evidence_example(
    example: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    evidence_by_quote: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Matches LLM-selected evidence against real evidence.

    Prefer evidence_id if the model returned it. Otherwise, match exact quote.
    """
    evidence_id = example.get("evidence_id")

    if isinstance(evidence_id, str) and evidence_id in evidence_by_id:
        return evidence_by_id[evidence_id]

    quote = example.get("quote")

    if isinstance(quote, str) and quote in evidence_by_quote:
        return evidence_by_quote[quote]

    return None


def normalize_sources(
    raw_sources: Any,
    sources_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Uses deterministic source metadata instead of trusting the LLM to recreate it.
    """
    return [
        {
            "source_id": item["source_id"],
            "url": item.get("url", ""),
            "title": item.get("title", ""),
        }
        for item in sources_summary
    ]


def normalize_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()

    return ""


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    normalized: list[str] = []

    for item in value:
        if isinstance(item, str) and item.strip():
            normalized.append(item.strip())

    return normalized


def parse_llm_json_response(raw_response: str) -> dict[str, Any]:
    """
    Parses JSON returned by the LLM.

    Handles clean JSON and common accidental code-fence formatting.
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