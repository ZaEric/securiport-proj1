from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.io import read_json, write_json, path_to_api_str
from src.api.run_artifacts import (
    build_curated_urls_input_json,
    build_search_api_input_json,
    build_synthetic_input_json,
    build_sources_json_from_synthetic,
)
from src.api.run_ids import generate_run_id
from src.api.schemas import (
    FromSyntheticRequest,
    FromCuratedUrlsRequest,
    FromSearchApiRequest,
    SourceCreationResponse,
)
from src.collection.curated import load_curated_person_entry
from src.collection.pipeline import build_sources_json_from_curated_urls, build_sources_json_from_search_api
from src.collection.search import build_search_query
from src.collection.url_filtering import filter_curated_urls

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
        run_dir=path_to_api_str(run_dir),
        input_path=path_to_api_str(input_path),
        sources_path=path_to_api_str(sources_path),
        num_sources=len(dataset["sources"]),
        status="created",
    )


@router.post("/from-curated-urls", response_model=SourceCreationResponse)
def create_sources_from_curated_urls(request: FromCuratedUrlsRequest) -> SourceCreationResponse:
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

    if not sources_json["sources"]:
        raise HTTPException(
            status_code=502,
            detail=(
                "No usable sources were created from curated URLs. "
                f"Fetch errors: {sources_json['metadata']['fetch_errors']}"
            ),
        )

    write_json(sources_path, sources_json)

    return SourceCreationResponse(
        run_id=run_id,
        target_name=curated_entry.person,
        run_dir=path_to_api_str(run_dir),
        input_path=path_to_api_str(input_path),
        sources_path=path_to_api_str(sources_path),
        num_sources=len(sources_json["sources"]),
        status="created",
    )


@router.post("/from-search-api", response_model=SourceCreationResponse)
def create_sources_from_search_api(request: FromSearchApiRequest) -> SourceCreationResponse:
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

    return SourceCreationResponse(
        run_id=run_id,
        target_name=request.target_name,
        run_dir=path_to_api_str(run_dir),
        input_path=path_to_api_str(input_path),
        sources_path=path_to_api_str(sources_path),
        num_sources=len(sources_json["sources"]),
        status="created",
    )
