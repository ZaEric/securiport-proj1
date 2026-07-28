from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.io import read_json, write_json, path_to_api_str
from src.api.schemas import (
    AggregationRequest,
    AggregationResponse,
    EvidenceClassifierRequest,
    EvidenceLLMRequest,
    EvidenceResponse,
    StatusResponse,
)
from src.nlp.models import EvidenceExtractionConfig
from src.nlp.pipeline import (
    process_chunks_for_evidence_llm,
    process_evidence_for_aggregation_llm,
)


router = APIRouter(prefix="/nlp", tags=["nlp"])


@router.post("/evidence-llm", response_model=EvidenceResponse)
def extract_evidence_llm(request: EvidenceLLMRequest) -> EvidenceResponse:
    run_dir = Path("data/runs") / request.run_id
    chunks_path = Path(request.chunks_path) if request.chunks_path else run_dir / "chunks.json"
    evidence_path = run_dir / "evidence.json"

    if not chunks_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Chunks file not found: {chunks_path}",
        )

    try:
        chunks_json = read_json(chunks_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read chunks file: {exc}",
        ) from exc

    required_keys = {"run_id", "target_name", "chunks"}
    missing_keys = required_keys - chunks_json.keys()

    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Chunks JSON missing required keys: {sorted(missing_keys)}",
        )

    if chunks_json["run_id"] != request.run_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Run ID mismatch: request run_id={request.run_id}, "
                f"chunks_json run_id={chunks_json['run_id']}"
            ),
        )

    default_config = EvidenceExtractionConfig()

    config = EvidenceExtractionConfig(
        model_name=request.model_name or default_config.model_name,
        temperature=default_config.temperature,
        seed=default_config.seed,
    )

    try:
        evidence_json = process_chunks_for_evidence_llm(
            chunks_json=chunks_json,
            config=config,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM evidence extraction failed: {exc}",
        ) from exc

    write_json(evidence_path, evidence_json)

    return EvidenceResponse(
        run_id=evidence_json["run_id"],
        target_name=evidence_json["target_name"],
        run_dir=path_to_api_str(run_dir),
        chunks_path=path_to_api_str(chunks_path),
        evidence_path=path_to_api_str(evidence_path),
        num_evidence=evidence_json["num_evidence"],
        status="created",
    )


@router.post("/evidence-classifier", response_model=StatusResponse)
def extract_evidence_classifier(request: EvidenceClassifierRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: classifier chunks -> sentiment labels/confidence -> evidence.json.",
    )


@router.post("/aggregate-llm", response_model=AggregationResponse)
def aggregate_llm(request: AggregationRequest) -> AggregationResponse:
    run_dir = Path("data/runs") / request.run_id
    evidence_path = Path(request.evidence_path) if request.evidence_path else run_dir / "evidence.json"
    aggregation_path = run_dir / "aggregation.json"

    if not evidence_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evidence file not found: {evidence_path}",
        )

    try:
        evidence_json = read_json(evidence_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read evidence file: {exc}",
        ) from exc

    required_keys = {"run_id", "target_name", "evidence"}
    missing_keys = required_keys - evidence_json.keys()

    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"Evidence JSON missing required keys: {sorted(missing_keys)}",
        )

    if evidence_json["run_id"] != request.run_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Run ID mismatch: request run_id={request.run_id}, "
                f"evidence_json run_id={evidence_json['run_id']}"
            ),
        )

    try:
        aggregation_json = process_evidence_for_aggregation_llm(
            evidence_json=evidence_json,
            mixed_evidence_threshold=request.mixed_evidence_threshold,
            baseline_tie_threshold=request.baseline_tie_threshold,
            minimum_evidence_count=request.minimum_evidence_count,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"LLM aggregation failed: {exc}",
        ) from exc

    write_json(aggregation_path, aggregation_json)

    return AggregationResponse(
        run_id=aggregation_json["run_id"],
        target_name=aggregation_json["target_name"],
        run_dir=path_to_api_str(run_dir),
        evidence_path=path_to_api_str(evidence_path),
        aggregation_path=path_to_api_str(aggregation_path),
        baseline_sentiment=aggregation_json["overall_result"]["baseline_sentiment"],
        num_sources=len(aggregation_json["source_results"]),
        num_evidence=aggregation_json["num_evidence"],
        status="created",
    )


@router.post("/aggregate-classifier", response_model=StatusResponse)
def aggregate_classifier(request: AggregationRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: classifier evidence.json -> confidence-weighted aggregation.json.",
    )