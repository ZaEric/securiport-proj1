from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.io import read_json, write_json
from src.api.run_artifacts import (
    build_synthetic_input_json,
    build_sources_json_from_synthetic,
)
from src.api.run_ids import generate_run_id
from src.api.schemas import (
    FromSyntheticRequest,
    FromCuratedUrlsRequest,
    FromSearchApiRequest,
    SourceCreationResponse,
    StatusResponse,
)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post("/from-synthetic", response_model=SourceCreationResponse)
def create_sources_from_synthetic(request: FromSyntheticRequest) -> SourceCreationResponse:
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

    sources_json = build_sources_json_from_synthetic(
        run_id=run_id,
        target_name=target_name,
        sources=dataset["sources"],
    )

    write_json(input_path, input_json)
    write_json(sources_path, sources_json)

    return SourceCreationResponse(
        run_id=run_id,
        target_name=target_name,
        run_dir=str(run_dir),
        input_path=str(input_path),
        sources_path=str(sources_path),
        num_sources=len(dataset["sources"]),
        status="created",
    )


@router.post("/from-curated-urls", response_model=StatusResponse)
def create_sources_from_curated_urls(request: FromCuratedUrlsRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: curated URLs -> scraping/text extraction -> sources.json.",
    )


@router.post("/from-search-api", response_model=StatusResponse)
def create_sources_from_search_api(request: FromSearchApiRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: search API -> URL filtering -> scraping/text extraction -> sources.json.",
    )