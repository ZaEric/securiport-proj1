from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.io import path_to_api_str, read_json, write_json
from src.api.run_artifacts import (
    build_curated_urls_input_json,
    build_search_api_input_json,
    build_sources_json_from_synthetic,
    build_synthetic_input_json,
)
from src.api.run_ids import generate_run_id
from src.api.schemas import (
    CuratedLLMRunRequest,
    FullPipelineResponse,
    SearchLLMRunRequest,
    SyntheticLLMRunRequest,
)
from src.collection.curated import load_curated_person_entry
from src.collection.pipeline import (
    build_sources_json_from_curated_urls,
    build_sources_json_from_search_api,
)
from src.collection.search import build_search_query
from src.collection.url_filtering import filter_curated_urls
from src.nlp.models import EvidenceExtractionConfig, ReportGenerationConfig
from src.nlp.pipeline import (
    process_chunks_for_evidence_llm,
    process_evidence_for_aggregation_llm,
    process_final_report_llm,
)
from src.processing.chunking import LLMChunkingConfig
from src.processing.pipeline import process_sources_for_nlp_llm

# Full-run defaults
DEFAULT_TARGET_CHUNK_WORDS = 3500
DEFAULT_MAX_CHUNK_WORDS = 4000
DEFAULT_OVERLAP_PARAGRAPHS = 1

DEFAULT_MIXED_EVIDENCE_THRESHOLD = 0.25
DEFAULT_BASELINE_TIE_THRESHOLD = 0.10
DEFAULT_MINIMUM_EVIDENCE_COUNT = 1


router = APIRouter(prefix="/pipeline", tags=["pipeline"])


@router.post("/run-synthetic-llm", response_model=FullPipelineResponse)
def run_synthetic_llm(request: SyntheticLLMRunRequest) -> FullPipelineResponse:
    synthetic_path = Path(request.synthetic_case_file)

    if not synthetic_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Synthetic case file not found: {synthetic_path}",
        )

    try:
        dataset: dict[str, Any] = read_json(synthetic_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read synthetic case file: {exc}",
        ) from exc

    required_keys = {"run_id", "target_name", "sources"}
    missing_keys = required_keys - dataset.keys()

    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Synthetic dataset missing required keys: {sorted(missing_keys)}",
        )

    target_name = dataset["target_name"]
    run_id = request.run_id or generate_run_id(target_name)

    run_dir = Path("data/runs") / run_id
    input_path = run_dir / "input.json"
    sources_path = run_dir / "sources.json"

    input_json = build_synthetic_input_json(
        run_id=run_id,
        target_name=target_name,
        synthetic_case_file=request.synthetic_case_file,
        expected_overall_sentiment=dataset.get("expected_overall_sentiment"),
        difficulty_tags=dataset.get("difficulty_tags", []),
        expected_failure_modes=dataset.get("expected_failure_modes", []),
        notes=dataset.get("case_notes"),
    )

    write_json(input_path, input_json)

    sources_json = build_sources_json_from_synthetic(
        run_id=run_id,
        target_name=target_name,
        sources=dataset["sources"],
    )

    write_json(sources_path, sources_json)

    try:
        chunks_json, evidence_json, aggregation_json, final_report_json = run_llm_analysis_stages(
            run_id=run_id,
            target_name=target_name,
            run_dir=run_dir,
            sources_json=sources_json,
            evidence_model_name=request.evidence_model_name,
            report_model_name=request.report_model_name,
            max_evidence_examples=request.max_evidence_examples,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Full synthetic LLM pipeline failed: {exc}",
        ) from exc

    return FullPipelineResponse(
        run_id=run_id,
        target_name=target_name,
        run_dir=path_to_api_str(run_dir),
        num_sources=len(sources_json["sources"]),
        num_chunks=len(chunks_json["chunks"]),
        num_evidence=evidence_json["num_evidence"],
        overall_sentiment=final_report_json["overall_sentiment"],
        status="created",
    )


@router.post("/run-curated-llm", response_model=FullPipelineResponse)
def run_curated_llm(request: CuratedLLMRunRequest) -> FullPipelineResponse:
    curated_path = Path(request.curated_urls_file)

    if not curated_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Curated URLs file not found: {curated_path}",
        )

    try:
        curated_entry = load_curated_person_entry(
            curated_urls_file=request.curated_urls_file,
            target_name=request.target_name,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read curated URLs file: {exc}",
        ) from exc

    run_id = request.run_id or generate_run_id(request.target_name)
    run_dir = Path("data/runs") / run_id
    input_path = run_dir / "input.json"
    sources_path = run_dir / "sources.json"

    filtered_urls = filter_curated_urls(
        urls=curated_entry.urls,
        max_urls=request.max_urls,
    )

    input_json = build_curated_urls_input_json(
        run_id=run_id,
        target_name=curated_entry.person,
        curated_urls_file=request.curated_urls_file,
        requested_urls=[item.url for item in filtered_urls],
        expected_overall_sentiment=curated_entry.expected_sentiment,
    )

    write_json(input_path, input_json)

    try:
        sources_json = build_sources_json_from_curated_urls(
            run_id=run_id,
            target_name=curated_entry.person,
            curated_urls_file=request.curated_urls_file,
            max_urls=request.max_urls,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create sources from curated URLs: {exc}",
        ) from exc

    write_json(sources_path, sources_json)

    if not sources_json["sources"]:
        raise HTTPException(
            status_code=502,
            detail=(
                "No usable sources were created from curated URLs. "
                f"Run artifacts were saved to {path_to_api_str(run_dir)}. "
                f"Fetch errors: {sources_json['metadata']['fetch_errors']}"
            ),
        )

    try:
        chunks_json, evidence_json, aggregation_json, final_report_json = run_llm_analysis_stages(
            run_id=run_id,
            target_name=curated_entry.person,
            run_dir=run_dir,
            sources_json=sources_json,
            evidence_model_name=request.evidence_model_name,
            report_model_name=request.report_model_name,
            max_evidence_examples=request.max_evidence_examples,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Full curated LLM pipeline failed: {exc}",
        ) from exc

    return FullPipelineResponse(
        run_id=run_id,
        target_name=curated_entry.person,
        run_dir=path_to_api_str(run_dir),
        num_sources=len(sources_json["sources"]),
        num_chunks=len(chunks_json["chunks"]),
        num_evidence=evidence_json["num_evidence"],
        overall_sentiment=final_report_json["overall_sentiment"],
        status="created",
    )


@router.post("/run-search-llm", response_model=FullPipelineResponse)
def run_search_llm(request: SearchLLMRunRequest) -> FullPipelineResponse:
    run_id = request.run_id or generate_run_id(request.target_name)
    run_dir = Path("data/runs") / run_id
    input_path = run_dir / "input.json"
    sources_path = run_dir / "sources.json"

    search_query = request.search_query or build_search_query(request.target_name)

    input_json = build_search_api_input_json(
        run_id=run_id,
        target_name=request.target_name,
        search_provider=request.search_provider,
        search_query=search_query,
        max_urls=request.max_urls,
    )

    write_json(input_path, input_json)

    try:
        sources_json = build_sources_json_from_search_api(
            run_id=run_id,
            target_name=request.target_name,
            search_provider=request.search_provider,
            search_query=search_query,
            max_urls=request.max_urls,
            run_dir=run_dir,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create sources from search API: {exc}",
        ) from exc

    write_json(sources_path, sources_json)

    if not sources_json["sources"]:
        raise HTTPException(
            status_code=502,
            detail=(
                "No usable sources were created from search API results. "
                f"Run artifacts were saved to {path_to_api_str(run_dir)}. "
                f"Fetch errors: {sources_json['metadata']['fetch_errors']}"
            ),
        )

    try:
        chunks_json, evidence_json, aggregation_json, final_report_json = run_llm_analysis_stages(
            run_id=run_id,
            target_name=request.target_name,
            run_dir=run_dir,
            sources_json=sources_json,
            evidence_model_name=request.evidence_model_name,
            report_model_name=request.report_model_name,
            max_evidence_examples=request.max_evidence_examples,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Full search LLM pipeline failed: {exc}",
        ) from exc

    return FullPipelineResponse(
        run_id=run_id,
        target_name=request.target_name,
        run_dir=path_to_api_str(run_dir),
        num_sources=len(sources_json["sources"]),
        num_chunks=len(chunks_json["chunks"]),
        num_evidence=evidence_json["num_evidence"],
        overall_sentiment=final_report_json["overall_sentiment"],
        status="created",
    )


def run_llm_analysis_stages(
    *,
    run_id: str,
    target_name: str,
    run_dir: Path,
    sources_json: dict[str, Any],
    evidence_model_name: str | None,
    report_model_name: str | None,
    max_evidence_examples: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    chunks_path = run_dir / "chunks.json"
    evidence_path = run_dir / "evidence.json"
    aggregation_path = run_dir / "aggregation.json"
    final_report_path = run_dir / "final_report.json"

    chunking_config = LLMChunkingConfig(
        target_chunk_words=DEFAULT_TARGET_CHUNK_WORDS,
        max_chunk_words=DEFAULT_MAX_CHUNK_WORDS,
        overlap_paragraphs=DEFAULT_OVERLAP_PARAGRAPHS,
    )

    chunks_json = process_sources_for_nlp_llm(
        raw_sources=sources_json["sources"],
        run_id=run_id,
        target_name=target_name,
        config=chunking_config,
    )

    write_json(chunks_path, chunks_json)

    default_evidence_config = EvidenceExtractionConfig()
    default_report_config = ReportGenerationConfig()

    evidence_config = EvidenceExtractionConfig(
        model_name=evidence_model_name or default_evidence_config.model_name,
        temperature=default_evidence_config.temperature,
        seed=default_evidence_config.seed,
    )

    report_config = ReportGenerationConfig(
        model_name=report_model_name or default_report_config.model_name,
        temperature=default_report_config.temperature,
        seed=default_report_config.seed,
    )

    evidence_json = process_chunks_for_evidence_llm(
        chunks_json=chunks_json,
        config=evidence_config,
    )

    write_json(evidence_path, evidence_json)

    aggregation_json = process_evidence_for_aggregation_llm(
        evidence_json=evidence_json,
        mixed_evidence_threshold=DEFAULT_MIXED_EVIDENCE_THRESHOLD,
        baseline_tie_threshold=DEFAULT_BASELINE_TIE_THRESHOLD,
        minimum_evidence_count=DEFAULT_MINIMUM_EVIDENCE_COUNT,
    )

    write_json(aggregation_path, aggregation_json)

    final_report_json = process_final_report_llm(
        sources_json=sources_json,
        evidence_json=evidence_json,
        aggregation_json=aggregation_json,
        config=report_config,
        max_evidence_examples=max_evidence_examples,
    )

    write_json(final_report_path, final_report_json)

    return chunks_json, evidence_json, aggregation_json, final_report_json