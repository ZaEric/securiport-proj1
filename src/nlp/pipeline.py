from typing import Any

from src.nlp.evidence_extraction import extract_evidence_from_chunk_llm
from src.nlp.llm_client import LLMClient, get_default_llm_client
from src.nlp.models import EvidenceExtractionConfig
from src.nlp.quote_verification import normalize_whitespace


def process_chunks_for_evidence_llm(
    chunks_json: dict[str, Any],
    config: EvidenceExtractionConfig | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    config = config or EvidenceExtractionConfig()
    llm_client = llm_client or get_default_llm_client()

    run_id = chunks_json["run_id"]
    target_name = chunks_json["target_name"]
    chunks = chunks_json["chunks"]

    raw_evidence_items: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_quotes = extract_evidence_from_chunk_llm(
            target_name=target_name,
            chunk_text=chunk["text"],
            llm_client=llm_client,
            config=config,
        )

        for extracted in chunk_quotes:
            raw_evidence_items.append(
                {
                    "source_id": chunk["source_id"],
                    "chunk_id": chunk["chunk_id"],
                    "quote": extracted.quote,
                    "sentiment": extracted.sentiment,
                    "confidence": None,
                }
            )

    deduped_evidence_items = dedupe_evidence_by_exact_quote(raw_evidence_items)

    final_evidence_items: list[dict[str, Any]] = []

    for index, item in enumerate(deduped_evidence_items, start=1):
        final_evidence_items.append(
            {
                "evidence_id": f"ev_{index:03d}",
                "source_id": item["source_id"],
                "chunk_id": item["chunk_id"],
                "quote": item["quote"],
                "sentiment": item["sentiment"],
                "confidence": item["confidence"],
            }
        )

    return {
        "run_id": run_id,
        "target_name": target_name,
        "evidence_extraction_method": config.evidence_extraction_method,
        "num_evidence": len(final_evidence_items),
        "evidence": final_evidence_items,
    }


def dedupe_evidence_by_exact_quote(
    evidence_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Removes duplicate evidence quotes caused by overlapping chunks.

    Uses normalized quote text as the dedupe key:
    - collapses repeated whitespace
    - lowercases for comparison
    - keeps the first occurrence
    """
    seen_quotes: set[str] = set()
    deduped_items: list[dict[str, Any]] = []

    for item in evidence_items:
        quote = item["quote"]
        dedupe_key = normalize_whitespace(quote).lower()

        if dedupe_key in seen_quotes:
            continue

        seen_quotes.add(dedupe_key)
        deduped_items.append(item)

    return deduped_items