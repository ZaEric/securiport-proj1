from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


RUNS_DIR = Path("data/runs")

REQUIRED_ARTIFACTS = [
    "input.json",
    "sources.json",
    "chunks.json",
    "evidence.json",
    "aggregation.json",
    "final_report.json",
]

VALID_SENTIMENTS = {"positive", "neutral", "negative"}


INPUT_COMMON_KEYS = {
    "run_id",
    "created_at",
    "target_name",
    "input_mode",
}

INPUT_MODE_KEYS = {
    "synthetic": INPUT_COMMON_KEYS
    | {
        "synthetic_case_file",
        "expected_overall_sentiment",
        "difficulty_tags",
        "expected_failure_modes",
        "notes",
    },
    "curated_urls": INPUT_COMMON_KEYS
    | {
        "curated_urls_file",
        "requested_urls",
        "expected_overall_sentiment",
    },
    "search_api": INPUT_COMMON_KEYS
    | {
        "search_provider",
        "search_query",
        "max_urls",
    },
}


SOURCES_TOP_LEVEL_KEYS = {
    "run_id",
    "target_name",
    "sources",
    "metadata",
}

SYNTHETIC_SOURCE_KEYS = {
    "source_id",
    "url",
    "title",
    "expected_source_sentiment",
    "article_text",
}

SCRAPED_SOURCE_KEYS = {
    "source_id",
    "url",
    "title",
    "retrieved_at",
    "article_text",
    "source_type",
    "notes",
    "status_code",
    "content_type",
}

SOURCES_METADATA_KEYS = {
    "synthetic": {
        "source_mode",
        "source_count",
    },
    "curated_urls": {
        "source_mode",
        "source_count",
        "requested_url_count",
        "failed_url_count",
        "blocked_url_count",
        "expected_overall_sentiment",
        "fetch_errors",
    },
    "search_api": {
        "source_mode",
        "search_provider",
        "search_query",
        "source_count",
        "requested_url_count",
        "failed_url_count",
        "blocked_url_count",
        "fetch_errors",
    },
}


CHUNKS_TOP_LEVEL_KEYS = {
    "run_id",
    "target_name",
    "chunks",
    "metadata",
}

CHUNK_ITEM_KEYS = {
    "chunk_id",
    "source_id",
    "text",
    "word_count",
}

CHUNKS_METADATA_KEYS = {
    "chunking_strategy",
    "target_chunk_words",
    "max_chunk_words",
    "overlap_paragraphs",
}


EVIDENCE_TOP_LEVEL_KEYS = {
    "run_id",
    "target_name",
    "evidence_extraction_method",
    "num_evidence",
    "num_evidence_extraction_issues",
    "llm_metadata",
    "evidence",
    "evidence_extraction_issues",
}

EVIDENCE_ITEM_KEYS = {
    "evidence_id",
    "source_id",
    "chunk_id",
    "quote",
    "sentiment",
    "confidence",
}

EVIDENCE_LLM_METADATA_KEYS = {
    "model_name",
    "temperature",
    "seed",
}


AGGREGATION_TOP_LEVEL_KEYS = {
    "run_id",
    "target_name",
    "aggregation_method",
    "num_evidence",
    "mixed_evidence_threshold",
    "minimum_evidence_count",
    "source_results",
    "overall_result",
}

AGGREGATION_SOURCE_RESULT_KEYS = {
    "source_id",
    "baseline_sentiment",
    "sentiment_counts",
    "negative_evidence_present",
    "mixed_evidence",
    "opposing_sentiment_ratio",
    "insufficient_evidence",
    "evidence_ids",
}

AGGREGATION_OVERALL_RESULT_KEYS = {
    "baseline_sentiment",
    "sentiment_counts",
    "source_baseline_sentiment_counts",
    "negative_evidence_present",
    "mixed_evidence",
    "opposing_sentiment_ratio",
    "insufficient_evidence",
}

SENTIMENT_COUNTS_KEYS = {
    "positive",
    "neutral",
    "negative",
}

SOURCE_BASELINE_SENTIMENT_COUNTS_KEYS = {
    "positive",
    "neutral",
    "negative",
}


FINAL_REPORT_TOP_LEVEL_KEYS = {
    "run_id",
    "target_name",
    "overall_sentiment",
    "llm_sentiment",
    "one_line_summary",
    "extended_summary",
    "key_findings",
    "justification",
    "evidence_examples",
    "negative_evidence",
    "sources",
    "limitations",
    "report_generation_issues",
    "llm_metadata",
}

FINAL_REPORT_EVIDENCE_EXAMPLE_KEYS = {
    "source_id",
    "quote",
    "sentiment",
    "why_selected",
}

FINAL_REPORT_NEGATIVE_SOURCE_KEYS = {
    "url",
    "title",
    "evidence",
}

FINAL_REPORT_NEGATIVE_EVIDENCE_KEYS = {
    "quote",
    "sentiment",
}

FINAL_REPORT_SOURCE_KEYS = {
    "source_id",
    "url",
    "title",
}

FINAL_REPORT_LLM_METADATA_KEYS = {
    "model_name",
    "temperature",
    "seed",
    "max_evidence_examples",
}


def main() -> None:
    if not RUNS_DIR.exists():
        raise SystemExit(f"Runs directory not found: {RUNS_DIR}")

    run_dirs = sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.name,
    )

    if not run_dirs:
        print(f"No runs found under {RUNS_DIR}")
        return

    summaries: list[dict[str, Any]] = []
    total_errors = 0

    for run_dir in run_dirs:
        summary, errors = validate_run(run_dir)
        summaries.append(summary)
        total_errors += len(errors)

        if errors:
            print(f"\nERRORS for {run_dir.name}:")
            for error in errors:
                print(f"  - {error}")

    print("\nRUN SUMMARY")
    print_run_summary_table(summaries)

    print("\nVALIDATION SUMMARY")
    print(f"Runs checked: {len(run_dirs)}")
    print(f"Runs passed: {sum(1 for item in summaries if item['status'] == 'PASS')}")
    print(f"Runs failed: {sum(1 for item in summaries if item['status'] == 'FAIL')}")
    print(f"Total errors: {total_errors}")

    if total_errors > 0:
        raise SystemExit(1)


def validate_run(run_dir: Path) -> tuple[dict[str, Any], list[str]]:
    run_id = run_dir.name
    errors: list[str] = []

    artifact_paths = {
        artifact_name: run_dir / artifact_name
        for artifact_name in REQUIRED_ARTIFACTS
    }

    for artifact_name, artifact_path in artifact_paths.items():
        if not artifact_path.exists():
            errors.append(f"Missing required artifact: {artifact_name}")

    loaded: dict[str, Any] = {}

    for artifact_name, artifact_path in artifact_paths.items():
        if artifact_path.exists():
            data = read_json(artifact_path)
            if data is None:
                errors.append(f"Failed to parse JSON: {artifact_name}")
            else:
                loaded[artifact_name] = data

    if len(loaded) != len(REQUIRED_ARTIFACTS):
        return build_partial_summary(run_id, loaded, errors), errors

    input_json = loaded["input.json"]
    sources_json = loaded["sources.json"]
    chunks_json = loaded["chunks.json"]
    evidence_json = loaded["evidence.json"]
    aggregation_json = loaded["aggregation.json"]
    final_report_json = loaded["final_report.json"]

    validate_input_json(input_json, errors)
    validate_sources_json(sources_json, input_json, errors)
    validate_chunks_json(chunks_json, input_json, errors)
    validate_evidence_json(evidence_json, input_json, errors)
    validate_aggregation_json(aggregation_json, input_json, evidence_json, errors)
    validate_final_report_json(
        final_report_json=final_report_json,
        input_json=input_json,
        evidence_json=evidence_json,
        aggregation_json=aggregation_json,
        errors=errors,
    )

    summary = build_summary(
        run_id=run_id,
        input_json=input_json,
        sources_json=sources_json,
        chunks_json=chunks_json,
        evidence_json=evidence_json,
        final_report_json=final_report_json,
        errors=errors,
    )

    return summary, errors


def validate_input_json(data: Any, errors: list[str]) -> None:
    if not isinstance(data, dict):
        errors.append("input.json must be an object")
        return

    input_mode = data.get("input_mode")

    if input_mode not in INPUT_MODE_KEYS:
        errors.append(f"input.json has invalid input_mode: {input_mode!r}")
        return

    check_exact_keys(
        data=data,
        expected_keys=INPUT_MODE_KEYS[input_mode],
        location="input.json",
        errors=errors,
    )

    check_type(data, "run_id", str, "input.json", errors)
    check_type(data, "created_at", str, "input.json", errors)
    check_type(data, "target_name", str, "input.json", errors)
    check_type(data, "input_mode", str, "input.json", errors)

    if input_mode == "synthetic":
        check_type(data, "synthetic_case_file", str, "input.json", errors)
        check_type(data, "expected_overall_sentiment", str, "input.json", errors)
        check_type(data, "difficulty_tags", list, "input.json", errors)
        check_type(data, "expected_failure_modes", list, "input.json", errors)
        check_type(data, "notes", str, "input.json", errors)

    elif input_mode == "curated_urls":
        check_type(data, "curated_urls_file", str, "input.json", errors)
        check_type(data, "requested_urls", list, "input.json", errors)
        check_type(data, "expected_overall_sentiment", str, "input.json", errors)

    elif input_mode == "search_api":
        check_type(data, "search_provider", str, "input.json", errors)
        check_type(data, "search_query", str, "input.json", errors)
        check_type(data, "max_urls", int, "input.json", errors)


def validate_sources_json(
    data: Any,
    input_json: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        errors.append("sources.json must be an object")
        return

    check_exact_keys(data, SOURCES_TOP_LEVEL_KEYS, "sources.json", errors)
    check_common_artifact_identity(data, input_json, "sources.json", errors)

    check_type(data, "sources", list, "sources.json", errors)
    check_type(data, "metadata", dict, "sources.json", errors)

    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        return

    source_mode = metadata.get("source_mode")

    if source_mode not in SOURCES_METADATA_KEYS:
        errors.append(f"sources.json metadata has invalid source_mode: {source_mode!r}")
        return

    input_mode = input_json.get("input_mode")
    if source_mode != input_mode:
        errors.append(
            f"sources.json metadata.source_mode {source_mode!r} does not match input_mode {input_mode!r}"
        )

    check_exact_keys(
        data=metadata,
        expected_keys=SOURCES_METADATA_KEYS[source_mode],
        location="sources.json.metadata",
        errors=errors,
    )

    check_type(metadata, "source_mode", str, "sources.json.metadata", errors)
    check_type(metadata, "source_count", int, "sources.json.metadata", errors)

    sources = data.get("sources")
    if isinstance(sources, list) and isinstance(metadata.get("source_count"), int):
        if metadata["source_count"] != len(sources):
            errors.append(
                f"sources.json metadata.source_count {metadata['source_count']} does not equal len(sources) {len(sources)}"
            )

    if source_mode == "synthetic":
        validate_source_items(
            sources=sources,
            expected_keys=SYNTHETIC_SOURCE_KEYS,
            location="sources.json.sources",
            errors=errors,
        )
    else:
        validate_source_items(
            sources=sources,
            expected_keys=SCRAPED_SOURCE_KEYS,
            location="sources.json.sources",
            errors=errors,
        )

    if source_mode == "curated_urls":
        check_type(metadata, "requested_url_count", int, "sources.json.metadata", errors)
        check_type(metadata, "failed_url_count", int, "sources.json.metadata", errors)
        check_type(metadata, "blocked_url_count", int, "sources.json.metadata", errors)
        check_type(metadata, "expected_overall_sentiment", str, "sources.json.metadata", errors)
        check_type(metadata, "fetch_errors", list, "sources.json.metadata", errors)
        validate_fetch_errors_light(metadata.get("fetch_errors"), errors)

    if source_mode == "search_api":
        check_type(metadata, "search_provider", str, "sources.json.metadata", errors)
        check_type(metadata, "search_query", str, "sources.json.metadata", errors)
        check_type(metadata, "requested_url_count", int, "sources.json.metadata", errors)
        check_type(metadata, "failed_url_count", int, "sources.json.metadata", errors)
        check_type(metadata, "blocked_url_count", int, "sources.json.metadata", errors)
        check_type(metadata, "fetch_errors", list, "sources.json.metadata", errors)
        validate_fetch_errors_light(metadata.get("fetch_errors"), errors)


def validate_source_items(
    sources: Any,
    expected_keys: set[str],
    location: str,
    errors: list[str],
) -> None:
    if not isinstance(sources, list):
        return

    for index, source in enumerate(sources):
        item_location = f"{location}[{index}]"

        if not isinstance(source, dict):
            errors.append(f"{item_location} must be an object")
            continue

        check_exact_keys(source, expected_keys, item_location, errors)

        for key in expected_keys:
            if key == "status_code":
                check_type(source, key, int, item_location, errors)
            else:
                check_type(source, key, str, item_location, errors)


def validate_fetch_errors_light(fetch_errors: Any, errors: list[str]) -> None:
    if not isinstance(fetch_errors, list):
        return

    for index, item in enumerate(fetch_errors):
        if not isinstance(item, dict):
            errors.append(f"sources.json.metadata.fetch_errors[{index}] must be an object")


def validate_chunks_json(
    data: Any,
    input_json: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        errors.append("chunks.json must be an object")
        return

    check_exact_keys(data, CHUNKS_TOP_LEVEL_KEYS, "chunks.json", errors)
    check_common_artifact_identity(data, input_json, "chunks.json", errors)

    check_type(data, "chunks", list, "chunks.json", errors)
    check_type(data, "metadata", dict, "chunks.json", errors)

    chunks = data.get("chunks")
    if isinstance(chunks, list):
        for index, chunk in enumerate(chunks):
            location = f"chunks.json.chunks[{index}]"

            if not isinstance(chunk, dict):
                errors.append(f"{location} must be an object")
                continue

            check_exact_keys(chunk, CHUNK_ITEM_KEYS, location, errors)
            check_type(chunk, "chunk_id", str, location, errors)
            check_type(chunk, "source_id", str, location, errors)
            check_type(chunk, "text", str, location, errors)
            check_type(chunk, "word_count", int, location, errors)

    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        check_exact_keys(metadata, CHUNKS_METADATA_KEYS, "chunks.json.metadata", errors)
        check_type(metadata, "chunking_strategy", str, "chunks.json.metadata", errors)
        check_type(metadata, "target_chunk_words", int, "chunks.json.metadata", errors)
        check_type(metadata, "max_chunk_words", int, "chunks.json.metadata", errors)
        check_type(metadata, "overlap_paragraphs", int, "chunks.json.metadata", errors)


def validate_evidence_json(
    data: Any,
    input_json: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        errors.append("evidence.json must be an object")
        return

    check_exact_keys(data, EVIDENCE_TOP_LEVEL_KEYS, "evidence.json", errors)
    check_common_artifact_identity(data, input_json, "evidence.json", errors)

    check_type(data, "evidence_extraction_method", str, "evidence.json", errors)
    check_type(data, "num_evidence", int, "evidence.json", errors)
    check_type(data, "num_evidence_extraction_issues", int, "evidence.json", errors)
    check_type(data, "llm_metadata", dict, "evidence.json", errors)
    check_type(data, "evidence", list, "evidence.json", errors)
    check_type(data, "evidence_extraction_issues", list, "evidence.json", errors)

    evidence = data.get("evidence")
    if isinstance(evidence, list):
        if data.get("num_evidence") != len(evidence):
            errors.append(
                f"evidence.json num_evidence {data.get('num_evidence')} does not equal len(evidence) {len(evidence)}"
            )

        for index, item in enumerate(evidence):
            location = f"evidence.json.evidence[{index}]"

            if not isinstance(item, dict):
                errors.append(f"{location} must be an object")
                continue

            check_exact_keys(item, EVIDENCE_ITEM_KEYS, location, errors)
            check_type(item, "evidence_id", str, location, errors)
            check_type(item, "source_id", str, location, errors)
            check_type(item, "chunk_id", str, location, errors)
            check_type(item, "quote", str, location, errors)
            check_type(item, "sentiment", str, location, errors)
            check_nullable_number(item, "confidence", location, errors)

            if item.get("sentiment") not in VALID_SENTIMENTS:
                errors.append(f"{location}.sentiment has invalid value: {item.get('sentiment')!r}")

    issues = data.get("evidence_extraction_issues")
    if isinstance(issues, list):
        if data.get("num_evidence_extraction_issues") != len(issues):
            errors.append(
                f"evidence.json num_evidence_extraction_issues {data.get('num_evidence_extraction_issues')} "
                f"does not equal len(evidence_extraction_issues) {len(issues)}"
            )

        for index, issue in enumerate(issues):
            if not isinstance(issue, dict):
                errors.append(f"evidence.json.evidence_extraction_issues[{index}] must be an object")

    llm_metadata = data.get("llm_metadata")
    if isinstance(llm_metadata, dict):
        check_exact_keys(
            llm_metadata,
            EVIDENCE_LLM_METADATA_KEYS,
            "evidence.json.llm_metadata",
            errors,
        )
        check_type(llm_metadata, "model_name", str, "evidence.json.llm_metadata", errors)
        check_number(llm_metadata, "temperature", "evidence.json.llm_metadata", errors)
        check_nullable_int(llm_metadata, "seed", "evidence.json.llm_metadata", errors)


def validate_aggregation_json(
    data: Any,
    input_json: dict[str, Any],
    evidence_json: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        errors.append("aggregation.json must be an object")
        return

    check_exact_keys(data, AGGREGATION_TOP_LEVEL_KEYS, "aggregation.json", errors)
    check_common_artifact_identity(data, input_json, "aggregation.json", errors)

    check_type(data, "aggregation_method", str, "aggregation.json", errors)
    check_type(data, "num_evidence", int, "aggregation.json", errors)
    check_number(data, "mixed_evidence_threshold", "aggregation.json", errors)
    check_type(data, "minimum_evidence_count", int, "aggregation.json", errors)
    check_type(data, "source_results", list, "aggregation.json", errors)
    check_type(data, "overall_result", dict, "aggregation.json", errors)

    if data.get("num_evidence") != evidence_json.get("num_evidence"):
        errors.append(
            f"aggregation.json num_evidence {data.get('num_evidence')} does not match evidence.json num_evidence {evidence_json.get('num_evidence')}"
        )

    source_results = data.get("source_results")
    if isinstance(source_results, list):
        for index, source_result in enumerate(source_results):
            location = f"aggregation.json.source_results[{index}]"

            if not isinstance(source_result, dict):
                errors.append(f"{location} must be an object")
                continue

            check_exact_keys(source_result, AGGREGATION_SOURCE_RESULT_KEYS, location, errors)
            check_type(source_result, "source_id", str, location, errors)
            check_type(source_result, "baseline_sentiment", str, location, errors)
            check_type(source_result, "sentiment_counts", dict, location, errors)
            check_type(source_result, "negative_evidence_present", bool, location, errors)
            check_type(source_result, "mixed_evidence", bool, location, errors)
            check_number(source_result, "opposing_sentiment_ratio", location, errors)
            check_type(source_result, "insufficient_evidence", bool, location, errors)
            check_type(source_result, "evidence_ids", list, location, errors)

            if source_result.get("baseline_sentiment") not in VALID_SENTIMENTS:
                errors.append(
                    f"{location}.baseline_sentiment has invalid value: {source_result.get('baseline_sentiment')!r}"
                )

            validate_sentiment_counts(
                source_result.get("sentiment_counts"),
                f"{location}.sentiment_counts",
                errors,
            )

    overall_result = data.get("overall_result")
    if isinstance(overall_result, dict):
        check_exact_keys(
            overall_result,
            AGGREGATION_OVERALL_RESULT_KEYS,
            "aggregation.json.overall_result",
            errors,
        )

        check_type(overall_result, "baseline_sentiment", str, "aggregation.json.overall_result", errors)
        check_type(overall_result, "sentiment_counts", dict, "aggregation.json.overall_result", errors)
        check_type(
            overall_result,
            "source_baseline_sentiment_counts",
            dict,
            "aggregation.json.overall_result",
            errors,
        )
        check_type(
            overall_result,
            "negative_evidence_present",
            bool,
            "aggregation.json.overall_result",
            errors,
        )
        check_type(overall_result, "mixed_evidence", bool, "aggregation.json.overall_result", errors)
        check_number(
            overall_result,
            "opposing_sentiment_ratio",
            "aggregation.json.overall_result",
            errors,
        )
        check_type(
            overall_result,
            "insufficient_evidence",
            bool,
            "aggregation.json.overall_result",
            errors,
        )

        if overall_result.get("baseline_sentiment") not in VALID_SENTIMENTS:
            errors.append(
                "aggregation.json.overall_result.baseline_sentiment has invalid value: "
                f"{overall_result.get('baseline_sentiment')!r}"
            )

        validate_sentiment_counts(
            overall_result.get("sentiment_counts"),
            "aggregation.json.overall_result.sentiment_counts",
            errors,
        )

        validate_source_baseline_sentiment_counts(
            overall_result.get("source_baseline_sentiment_counts"),
            "aggregation.json.overall_result.source_baseline_sentiment_counts",
            errors,
        )

        counted_sentiments = count_evidence_sentiments(evidence_json)
        aggregation_counts = overall_result.get("sentiment_counts")

        if isinstance(aggregation_counts, dict):
            for sentiment in SENTIMENT_COUNTS_KEYS:
                if aggregation_counts.get(sentiment) != counted_sentiments.get(sentiment, 0):
                    errors.append(
                        f"aggregation overall sentiment_counts.{sentiment}={aggregation_counts.get(sentiment)} "
                        f"does not match evidence count {counted_sentiments.get(sentiment, 0)}"
                    )


def validate_final_report_json(
    *,
    final_report_json: Any,
    input_json: dict[str, Any],
    evidence_json: dict[str, Any],
    aggregation_json: dict[str, Any],
    errors: list[str],
) -> None:
    if not isinstance(final_report_json, dict):
        errors.append("final_report.json must be an object")
        return

    check_exact_keys(
        final_report_json,
        FINAL_REPORT_TOP_LEVEL_KEYS,
        "final_report.json",
        errors,
    )
    check_common_artifact_identity(final_report_json, input_json, "final_report.json", errors)

    check_type(final_report_json, "overall_sentiment", str, "final_report.json", errors)
    check_type(final_report_json, "llm_sentiment", str, "final_report.json", errors)
    check_type(final_report_json, "one_line_summary", str, "final_report.json", errors)
    check_type(final_report_json, "extended_summary", str, "final_report.json", errors)
    check_type(final_report_json, "key_findings", list, "final_report.json", errors)
    check_type(final_report_json, "justification", str, "final_report.json", errors)
    check_type(final_report_json, "evidence_examples", list, "final_report.json", errors)
    check_type(final_report_json, "negative_evidence", list, "final_report.json", errors)
    check_type(final_report_json, "sources", list, "final_report.json", errors)
    check_type(final_report_json, "limitations", list, "final_report.json", errors)
    check_type(final_report_json, "report_generation_issues", list, "final_report.json", errors)
    check_type(final_report_json, "llm_metadata", dict, "final_report.json", errors)

    if final_report_json.get("overall_sentiment") not in VALID_SENTIMENTS:
        errors.append(
            f"final_report.json overall_sentiment has invalid value: {final_report_json.get('overall_sentiment')!r}"
        )

    if final_report_json.get("llm_sentiment") not in VALID_SENTIMENTS:
        errors.append(
            f"final_report.json llm_sentiment has invalid value: {final_report_json.get('llm_sentiment')!r}"
        )

    aggregation_sentiment = (
        aggregation_json.get("overall_result", {}).get("baseline_sentiment")
        if isinstance(aggregation_json, dict)
        else None
    )

    if final_report_json.get("overall_sentiment") != aggregation_sentiment:
        errors.append(
            f"final_report overall_sentiment {final_report_json.get('overall_sentiment')!r} "
            f"does not match aggregation baseline_sentiment {aggregation_sentiment!r}"
        )

    if final_report_json.get("llm_sentiment") != final_report_json.get("overall_sentiment"):
        errors.append(
            f"final_report llm_sentiment {final_report_json.get('llm_sentiment')!r} "
            f"does not match overall_sentiment {final_report_json.get('overall_sentiment')!r}"
        )

    validate_string_list(final_report_json.get("key_findings"), "final_report.json.key_findings", errors)
    validate_string_list(final_report_json.get("limitations"), "final_report.json.limitations", errors)

    evidence_examples = final_report_json.get("evidence_examples")
    if isinstance(evidence_examples, list):
        for index, item in enumerate(evidence_examples):
            location = f"final_report.json.evidence_examples[{index}]"

            if not isinstance(item, dict):
                errors.append(f"{location} must be an object")
                continue

            check_exact_keys(item, FINAL_REPORT_EVIDENCE_EXAMPLE_KEYS, location, errors)
            check_type(item, "source_id", str, location, errors)
            check_type(item, "quote", str, location, errors)
            check_type(item, "sentiment", str, location, errors)
            check_type(item, "why_selected", str, location, errors)

            if item.get("sentiment") not in VALID_SENTIMENTS:
                errors.append(f"{location}.sentiment has invalid value: {item.get('sentiment')!r}")

    negative_evidence = final_report_json.get("negative_evidence")
    if isinstance(negative_evidence, list):
        for source_index, source_group in enumerate(negative_evidence):
            source_location = f"final_report.json.negative_evidence[{source_index}]"

            if not isinstance(source_group, dict):
                errors.append(f"{source_location} must be an object")
                continue

            check_exact_keys(source_group, FINAL_REPORT_NEGATIVE_SOURCE_KEYS, source_location, errors)
            check_type(source_group, "url", str, source_location, errors)
            check_type(source_group, "title", str, source_location, errors)
            check_type(source_group, "evidence", list, source_location, errors)

            evidence_items = source_group.get("evidence")
            if isinstance(evidence_items, list):
                for evidence_index, item in enumerate(evidence_items):
                    item_location = f"{source_location}.evidence[{evidence_index}]"

                    if not isinstance(item, dict):
                        errors.append(f"{item_location} must be an object")
                        continue

                    check_exact_keys(
                        item,
                        FINAL_REPORT_NEGATIVE_EVIDENCE_KEYS,
                        item_location,
                        errors,
                    )
                    check_type(item, "quote", str, item_location, errors)
                    check_type(item, "sentiment", str, item_location, errors)

                    if item.get("sentiment") != "negative":
                        errors.append(f"{item_location}.sentiment must be 'negative'")

    sources = final_report_json.get("sources")
    if isinstance(sources, list):
        for index, source in enumerate(sources):
            location = f"final_report.json.sources[{index}]"

            if not isinstance(source, dict):
                errors.append(f"{location} must be an object")
                continue

            check_exact_keys(source, FINAL_REPORT_SOURCE_KEYS, location, errors)
            check_type(source, "source_id", str, location, errors)
            check_type(source, "url", str, location, errors)
            check_type(source, "title", str, location, errors)

    report_issues = final_report_json.get("report_generation_issues")
    if isinstance(report_issues, list):
        for index, issue in enumerate(report_issues):
            if not isinstance(issue, (str, dict)):
                errors.append(
                    f"final_report.json.report_generation_issues[{index}] must be a string or object"
                )

    llm_metadata = final_report_json.get("llm_metadata")
    if isinstance(llm_metadata, dict):
        check_exact_keys(
            llm_metadata,
            FINAL_REPORT_LLM_METADATA_KEYS,
            "final_report.json.llm_metadata",
            errors,
        )
        check_type(llm_metadata, "model_name", str, "final_report.json.llm_metadata", errors)
        check_number(llm_metadata, "temperature", "final_report.json.llm_metadata", errors)
        check_nullable_int(llm_metadata, "seed", "final_report.json.llm_metadata", errors)
        check_type(llm_metadata, "max_evidence_examples", int, "final_report.json.llm_metadata", errors)


def check_common_artifact_identity(
    data: dict[str, Any],
    input_json: dict[str, Any],
    location: str,
    errors: list[str],
) -> None:
    check_type(data, "run_id", str, location, errors)
    check_type(data, "target_name", str, location, errors)

    if data.get("run_id") != input_json.get("run_id"):
        errors.append(
            f"{location} run_id {data.get('run_id')!r} does not match input.json run_id {input_json.get('run_id')!r}"
        )

    if data.get("target_name") != input_json.get("target_name"):
        errors.append(
            f"{location} target_name {data.get('target_name')!r} does not match input.json target_name {input_json.get('target_name')!r}"
        )


def check_exact_keys(
    data: dict[str, Any],
    expected_keys: set[str],
    location: str,
    errors: list[str],
) -> None:
    actual_keys = set(data.keys())

    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)

    if missing:
        errors.append(f"{location} missing keys: {missing}")

    if extra:
        errors.append(f"{location} has unexpected keys: {extra}")


def check_type(
    data: dict[str, Any],
    key: str,
    expected_type: type,
    location: str,
    errors: list[str],
) -> None:
    if key not in data:
        return

    value = data[key]

    if expected_type is int:
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected_type is bool:
        valid = isinstance(value, bool)
    else:
        valid = isinstance(value, expected_type)

    if not valid:
        errors.append(
            f"{location}.{key} expected {expected_type.__name__}, got {type(value).__name__}"
        )


def check_number(
    data: dict[str, Any],
    key: str,
    location: str,
    errors: list[str],
) -> None:
    if key not in data:
        return

    value = data[key]

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{location}.{key} expected number, got {type(value).__name__}")


def check_nullable_number(
    data: dict[str, Any],
    key: str,
    location: str,
    errors: list[str],
) -> None:
    if key not in data:
        return

    value = data[key]

    if value is None:
        return

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(
            f"{location}.{key} expected number or null, got {type(value).__name__}"
        )


def check_nullable_int(
    data: dict[str, Any],
    key: str,
    location: str,
    errors: list[str],
) -> None:
    if key not in data:
        return

    value = data[key]

    if value is None:
        return

    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(
            f"{location}.{key} expected int or null, got {type(value).__name__}"
        )


def validate_sentiment_counts(
    data: Any,
    location: str,
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        return

    check_exact_keys(data, SENTIMENT_COUNTS_KEYS, location, errors)

    for key in SENTIMENT_COUNTS_KEYS:
        check_type(data, key, int, location, errors)


def validate_source_baseline_sentiment_counts(
    data: Any,
    location: str,
    errors: list[str],
) -> None:
    if not isinstance(data, dict):
        return

    check_exact_keys(data, SOURCE_BASELINE_SENTIMENT_COUNTS_KEYS, location, errors)

    for key in SOURCE_BASELINE_SENTIMENT_COUNTS_KEYS:
        check_type(data, key, int, location, errors)


def validate_string_list(
    data: Any,
    location: str,
    errors: list[str],
) -> None:
    if not isinstance(data, list):
        return

    for index, item in enumerate(data):
        if not isinstance(item, str):
            errors.append(f"{location}[{index}] expected str, got {type(item).__name__}")


def count_evidence_sentiments(evidence_json: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()

    evidence = evidence_json.get("evidence", [])

    if not isinstance(evidence, list):
        return counter

    for item in evidence:
        if isinstance(item, dict):
            sentiment = item.get("sentiment")
            if isinstance(sentiment, str):
                counter[sentiment] += 1

    for sentiment in SENTIMENT_COUNTS_KEYS:
        counter.setdefault(sentiment, 0)

    return counter


def build_summary(
    *,
    run_id: str,
    input_json: dict[str, Any],
    sources_json: dict[str, Any],
    chunks_json: dict[str, Any],
    evidence_json: dict[str, Any],
    final_report_json: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    source_metadata = sources_json.get("metadata", {})
    sentiment_counts = count_evidence_sentiments(evidence_json)

    evidence_issues = evidence_json.get("num_evidence_extraction_issues", 0)
    report_issues = final_report_json.get("report_generation_issues", [])

    return {
        "run_folder": run_id,
        "artifact_run_id": input_json.get("run_id", "N/A"),
        "input_mode": input_json.get("input_mode", "N/A"),
        "sources": source_metadata.get("source_count", "N/A")
        if isinstance(source_metadata, dict)
        else "N/A",
        "failed_urls": source_metadata.get("failed_url_count", 0)
        if isinstance(source_metadata, dict)
        else "N/A",
        "chunks": len(chunks_json.get("chunks", []))
        if isinstance(chunks_json.get("chunks"), list)
        else "N/A",
        "evidence": evidence_json.get("num_evidence", "N/A"),
        "positive": sentiment_counts["positive"],
        "neutral": sentiment_counts["neutral"],
        "negative": sentiment_counts["negative"],
        "overall": final_report_json.get("overall_sentiment", "N/A"),
        "llm": final_report_json.get("llm_sentiment", "N/A"),
        "issues": safe_issue_count(evidence_issues, report_issues),
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
    }


def build_partial_summary(
    run_id: str,
    loaded: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    input_json = loaded.get("input.json", {})
    sources_json = loaded.get("sources.json", {})
    chunks_json = loaded.get("chunks.json", {})
    evidence_json = loaded.get("evidence.json", {})
    final_report_json = loaded.get("final_report.json", {})

    if not isinstance(input_json, dict):
        input_json = {}
    if not isinstance(sources_json, dict):
        sources_json = {}
    if not isinstance(chunks_json, dict):
        chunks_json = {}
    if not isinstance(evidence_json, dict):
        evidence_json = {}
    if not isinstance(final_report_json, dict):
        final_report_json = {}

    return build_summary(
        run_id=run_id,
        input_json=input_json,
        sources_json=sources_json,
        chunks_json=chunks_json,
        evidence_json=evidence_json,
        final_report_json=final_report_json,
        errors=errors,
    )


def safe_issue_count(evidence_issues: Any, report_issues: Any) -> int | str:
    total = 0

    if isinstance(evidence_issues, int):
        total += evidence_issues
    else:
        return "N/A"

    if isinstance(report_issues, list):
        total += len(report_issues)
    else:
        return "N/A"

    return total


def print_run_summary_table(summaries: list[dict[str, Any]]) -> None:
    columns = [
        "run_folder",
        "artifact_run_id",
        "input_mode",
        "sources",
        "failed_urls",
        "chunks",
        "evidence",
        "positive",
        "neutral",
        "negative",
        "overall",
        "llm",
        "issues",
        "status",
        "error_count",
    ]

    widths = {
        column: max(
            len(column),
            max(len(str(summary.get(column, ""))) for summary in summaries),
        )
        for column in columns
    }

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)

    print(header)
    print(divider)

    for summary in summaries:
        row = " | ".join(
            str(summary.get(column, "")).ljust(widths[column])
            for column in columns
        )
        print(row)


def read_json(path: Path) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


if __name__ == "__main__":
    main()