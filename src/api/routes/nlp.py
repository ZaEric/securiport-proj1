from fastapi import APIRouter

from src.api.schemas import (
    AggregationRequest,
    EvidenceClassifierRequest,
    EvidenceLLMRequest,
    StatusResponse,
)


router = APIRouter(prefix="/nlp", tags=["nlp"])


@router.post("/evidence-llm", response_model=StatusResponse)
def evidence_llm(request: EvidenceLLMRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: chunks.json -> LLM evidence extraction -> evidence.json.",
    )


@router.post("/evidence-classifier", response_model=StatusResponse)
def evidence_classifier(request: EvidenceClassifierRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: chunks.json -> classifier evidence extraction -> evidence.json.",
    )


@router.post("/aggregate-llm", response_model=StatusResponse)
def aggregate_llm(request: AggregationRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: evidence.json -> label-count LLM aggregation -> aggregation.json.",
    )


@router.post("/aggregate-classifier", response_model=StatusResponse)
def aggregate_classifier(request: AggregationRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: evidence.json -> confidence-weighted classifier aggregation -> aggregation.json.",
    )