from collections import defaultdict
from typing import Any


SENTIMENT_LABELS = ("positive", "neutral", "negative")
DEFAULT_MIXED_EVIDENCE_THRESHOLD = 0.25
DEFAULT_BASELINE_TIE_THRESHOLD = 0.10
DEFAULT_MINIMUM_EVIDENCE_COUNT = 1


def aggregate_evidence_llm(
    evidence_json: dict[str, Any],
    mixed_evidence_threshold: float = DEFAULT_MIXED_EVIDENCE_THRESHOLD,
    baseline_tie_threshold: float = DEFAULT_BASELINE_TIE_THRESHOLD,
    minimum_evidence_count: int = DEFAULT_MINIMUM_EVIDENCE_COUNT,
) -> dict[str, Any]:
    """
    Aggregates LLM-extracted evidence into source-level and overall summaries.

    This creates a deterministic baseline summary. Final nuanced interpretation
    is left for final report generation.
    """
    run_id = evidence_json["run_id"]
    target_name = evidence_json["target_name"]
    evidence_items = evidence_json["evidence"]

    evidence_by_source = group_evidence_by_source(evidence_items)

    source_results = [
        aggregate_source_evidence(
            source_id=source_id,
            evidence_items=source_evidence,
            mixed_evidence_threshold=mixed_evidence_threshold,
            baseline_tie_threshold=baseline_tie_threshold,
            minimum_evidence_count=minimum_evidence_count,
        )
        for source_id, source_evidence in evidence_by_source.items()
    ]

    overall_result = aggregate_overall_evidence(
        source_results=source_results,
        evidence_items=evidence_items,
        mixed_evidence_threshold=mixed_evidence_threshold,
        baseline_tie_threshold=baseline_tie_threshold,
        minimum_evidence_count=minimum_evidence_count,
    )

    return {
        "run_id": run_id,
        "target_name": target_name,
        "aggregation_method": "label_count_summary",
        "num_evidence": len(evidence_items),
        "mixed_evidence_threshold": mixed_evidence_threshold,
        "baseline_tie_threshold": baseline_tie_threshold,
        "minimum_evidence_count": minimum_evidence_count,
        "source_results": source_results,
        "overall_result": overall_result,
    }


def group_evidence_by_source(
    evidence_items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Groups evidence snippets by source_id.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in evidence_items:
        grouped[item["source_id"]].append(item)

    return dict(grouped)


def aggregate_source_evidence(
    source_id: str,
    evidence_items: list[dict[str, Any]],
    mixed_evidence_threshold: float,
    baseline_tie_threshold: float,
    minimum_evidence_count: int,
) -> dict[str, Any]:
    """
    Creates one source-level aggregation result.

    mixed_evidence is true only when positive and negative evidence both exist
    in a meaningful ratio.
    """
    sentiment_counts = count_sentiments(evidence_items)
    baseline_sentiment = decide_baseline_sentiment(
        sentiment_counts=sentiment_counts,
        baseline_tie_threshold=baseline_tie_threshold,
    )
    opposing_sentiment_ratio = calculate_opposing_sentiment_ratio(
        sentiment_counts=sentiment_counts,
        baseline_sentiment=baseline_sentiment,
    )

    return {
        "source_id": source_id,
        "baseline_sentiment": baseline_sentiment,
        "sentiment_counts": sentiment_counts,
        "mixed_evidence": opposing_sentiment_ratio >= mixed_evidence_threshold,
        "opposing_sentiment_ratio": opposing_sentiment_ratio,
        "insufficient_evidence": is_insufficient_evidence(
            total_evidence_count=len(evidence_items),
            minimum_evidence_count=minimum_evidence_count,
        ),
        "evidence_ids": [item["evidence_id"] for item in evidence_items],
    }


def aggregate_overall_evidence(
    source_results: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    mixed_evidence_threshold: float,
    baseline_tie_threshold: float,
    minimum_evidence_count: int,
) -> dict[str, Any]:
    """
    Creates the overall aggregation result across all sources.
    """
    sentiment_counts = count_sentiments(evidence_items)
    source_baseline_sentiment_counts = count_source_baseline_sentiments(source_results)

    baseline_sentiment = decide_baseline_sentiment(
        sentiment_counts=sentiment_counts,
        baseline_tie_threshold=baseline_tie_threshold,
    )
    opposing_sentiment_ratio = calculate_opposing_sentiment_ratio(
        sentiment_counts=sentiment_counts,
        baseline_sentiment=baseline_sentiment,
    )

    return {
        "baseline_sentiment": baseline_sentiment,
        "sentiment_counts": sentiment_counts,
        "source_baseline_sentiment_counts": source_baseline_sentiment_counts,
        "mixed_evidence": opposing_sentiment_ratio >= mixed_evidence_threshold,
        "opposing_sentiment_ratio": opposing_sentiment_ratio,
        "insufficient_evidence": is_insufficient_evidence(
            total_evidence_count=len(evidence_items),
            minimum_evidence_count=minimum_evidence_count,
        ),
    }


def count_sentiments(evidence_items: list[dict[str, Any]]) -> dict[str, int]:
    """
    Counts quote-level sentiment labels.
    """
    counts = {label: 0 for label in SENTIMENT_LABELS}

    for item in evidence_items:
        sentiment = item.get("sentiment")

        if sentiment in counts:
            counts[sentiment] += 1

    return counts


def count_source_baseline_sentiments(
    source_results: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Counts source-level baseline sentiment labels.
    """
    counts = {label: 0 for label in SENTIMENT_LABELS}

    for result in source_results:
        baseline_sentiment = result.get("baseline_sentiment")

        if baseline_sentiment in counts:
            counts[baseline_sentiment] += 1

    return counts


def decide_baseline_sentiment(
    sentiment_counts: dict[str, int],
    baseline_tie_threshold: float,
) -> str:
    """
    Chooses a simple baseline sentiment.

    Positive/negative evidence takes priority over neutral background evidence.
    If positive and negative are close enough, return neutral.

    Example with threshold 0.10:
    18 positive vs 19 negative -> neutral because the difference is small.
    """
    positive_count = sentiment_counts["positive"]
    negative_count = sentiment_counts["negative"]
    neutral_count = sentiment_counts["neutral"]

    non_neutral_total = positive_count + negative_count

    if non_neutral_total == 0:
        return "neutral" if neutral_count > 0 else "neutral"

    difference_ratio = abs(positive_count - negative_count) / non_neutral_total

    if difference_ratio <= baseline_tie_threshold:
        return "neutral"

    if positive_count > negative_count:
        return "positive"

    return "negative"


def calculate_opposing_sentiment_ratio(
    sentiment_counts: dict[str, int],
    baseline_sentiment: str,
) -> float:
    """
    Calculates how much non-neutral evidence opposes the baseline sentiment.

    Example:
    14 negative, 1 positive -> baseline negative -> ratio = 1 / 15 = 0.067
    """
    positive_count = sentiment_counts["positive"]
    negative_count = sentiment_counts["negative"]
    non_neutral_total = positive_count + negative_count

    if non_neutral_total == 0:
        return 0.0

    if baseline_sentiment == "positive":
        return round(negative_count / non_neutral_total, 3)

    if baseline_sentiment == "negative":
        return round(positive_count / non_neutral_total, 3)

    # If baseline is neutral due to near-tie, conflicting evidence is balanced.
    if positive_count > 0 and negative_count > 0:
        return round(min(positive_count, negative_count) / non_neutral_total, 3)

    return 0.0


def is_insufficient_evidence(
    total_evidence_count: int,
    minimum_evidence_count: int,
) -> bool:
    """
    MVP rule: insufficient evidence if fewer than minimum_evidence_count snippets exist.
    """
    return total_evidence_count < minimum_evidence_count