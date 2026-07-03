from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.io import read_json, write_json, path_to_api_str
from src.api.schemas import (
    ChunkClassifierRequest,
    ChunkLLMRequest,
    ChunkResponse,
    StatusResponse,
)
from src.processing.chunking import LLMChunkingConfig
from src.processing.pipeline import process_sources_for_nlp_llm


router = APIRouter(prefix="/processing", tags=["processing"])


@router.post("/chunk-llm", response_model=ChunkResponse)
def chunk_llm(request: ChunkLLMRequest) -> ChunkResponse:
    run_dir = Path("data/runs") / request.run_id
    sources_path = Path(request.sources_path) if request.sources_path else run_dir / "sources.json"
    chunks_path = run_dir / "chunks.json"

    if not sources_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"sources.json not found: {sources_path}",
        )

    try:
        sources_json: dict[str, Any] = read_json(sources_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read sources.json: {exc}",
        ) from exc

    required_keys = {"run_id", "target_name", "sources"}
    missing_keys = required_keys - sources_json.keys()

    if missing_keys:
        raise HTTPException(
            status_code=400,
            detail=f"sources.json missing required keys: {sorted(missing_keys)}",
        )

    config = LLMChunkingConfig(
        target_chunk_words=request.target_chunk_words,
        max_chunk_words=request.max_chunk_words,
        overlap_paragraphs=request.overlap_paragraphs,
    )

    chunks_json = process_sources_for_nlp_llm(
        raw_sources=sources_json["sources"],
        run_id=request.run_id,
        target_name=sources_json["target_name"],
        config=config,
    )

    write_json(chunks_path, chunks_json)

    return ChunkResponse(
        run_id=request.run_id,
        target_name=sources_json["target_name"],
        run_dir=path_to_api_str(run_dir),
        sources_path=path_to_api_str(sources_path),
        chunks_path=path_to_api_str(chunks_path),
        num_sources=len(sources_json["sources"]),
        num_chunks=len(chunks_json["chunks"]),
        status="created",
    )


@router.post("/chunk-classifier", response_model=StatusResponse)
def chunk_classifier(request: ChunkClassifierRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: sources.json -> sentence/window chunks.json for classifier evidence extraction.",
    )