from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


CURRENT_RUN_ID = "2026-06-29_1642_john_doe"
CURRENT_RUN_DIR = f"data/runs/{CURRENT_RUN_ID}"


class InputMode(str, Enum):
    synthetic = "synthetic"
    curated_urls = "curated_urls"
    search_api = "search_api"


class AnalysisMode(str, Enum):
    llm = "llm"
    classifier = "classifier"


class StatusResponse(BaseModel):
    status: str
    message: str


# -------------------------
# Source creation requests
# -------------------------

class FromSyntheticRequest(BaseModel):
    synthetic_case_file: str
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "synthetic_case_file": "data/synthetic/passengers/synthetic_negative_001.json",
                "run_id": CURRENT_RUN_ID,
            }
        }
    )


class FromCuratedUrlsRequest(BaseModel):
    target_name: str
    curated_urls_file: str
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_name": "John Doe",
                "curated_urls_file": "data/curated/curated_urls.json",
                "run_id": CURRENT_RUN_ID,
            }
        }
    )


class FromSearchApiRequest(BaseModel):
    target_name: str
    search_provider: str = "bing"
    search_query: str | None = Field(
        default=None,
        description="Optional search query. If omitted, one can be generated from target_name.",
    )
    max_urls: int = Field(default=10, ge=1)
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_name": "John Doe",
                "search_provider": "bing",
                "search_query": "\"John Doe\" news",
                "max_urls": 10,
                "run_id": CURRENT_RUN_ID,
            }
        }
    )


class SourceCreationResponse(BaseModel):
    run_id: str
    target_name: str
    run_dir: str
    input_path: str
    sources_path: str
    num_sources: int
    status: str


# -------------------------
# Chunking requests
# -------------------------

class ChunkLLMRequest(BaseModel):
    run_id: str
    sources_path: str | None = Field(
        default=None,
        description="Optional path to sources.json. Defaults to data/runs/<run_id>/sources.json.",
    )
    target_chunk_words: int = Field(default=600, ge=1)
    max_chunk_words: int = Field(default=700, ge=1)
    overlap_paragraphs: int = Field(default=1, ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "sources_path": None,
                "target_chunk_words": 600,
                "max_chunk_words": 700,
                "overlap_paragraphs": 1,
            }
        }
    )


class ChunkClassifierRequest(BaseModel):
    run_id: str
    sources_path: str | None = Field(
        default=None,
        description="Optional path to sources.json. Defaults to data/runs/<run_id>/sources.json.",
    )
    window_before: int = Field(default=1, ge=0)
    window_after: int = Field(default=1, ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "sources_path": None,
                "window_before": 1,
                "window_after": 1,
            }
        }
    )


class ChunkResponse(BaseModel):
    run_id: str
    target_name: str
    run_dir: str
    sources_path: str
    chunks_path: str
    num_sources: int
    num_chunks: int
    status: str


# -------------------------
# Evidence extraction requests
# -------------------------

class EvidenceLLMRequest(BaseModel):
    run_id: str
    chunks_path: str | None = Field(
        default=None,
        description="Optional path to chunks.json. Defaults to data/runs/<run_id>/chunks.json.",
    )
    model_name: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "chunks_path": None,
                "model_name": None,
            }
        }
    )


class EvidenceClassifierRequest(BaseModel):
    run_id: str
    chunks_path: str | None = Field(
        default=None,
        description="Optional path to chunks.json. Defaults to data/runs/<run_id>/chunks.json.",
    )
    model_name: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "chunks_path": None,
                "model_name": None,
            }
        }
    )

class EvidenceResponse(BaseModel):
    run_id: str
    target_name: str
    run_dir: str
    chunks_path: str
    evidence_path: str
    num_evidence: int
    status: str


# -------------------------
# Aggregation requests
# -------------------------

class AggregationRequest(BaseModel):
    run_id: str
    evidence_path: str | None = Field(
        default=None,
        description="Optional path to evidence.json. Defaults to data/runs/<run_id>/evidence.json.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "evidence_path": None,
            }
        }
    )


# -------------------------
# Final report request
# -------------------------

class ReportGenerateRequest(BaseModel):
    run_id: str
    sources_path: str | None = Field(
        default=None,
        description="Optional path to sources.json. Defaults to data/runs/<run_id>/sources.json.",
    )
    evidence_path: str | None = Field(
        default=None,
        description="Optional path to evidence.json. Defaults to data/runs/<run_id>/evidence.json.",
    )
    aggregation_path: str | None = Field(
        default=None,
        description="Optional path to aggregation.json. Defaults to data/runs/<run_id>/aggregation.json.",
    )
    model_name: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "sources_path": None,
                "evidence_path": None,
                "aggregation_path": None,
                "model_name": None,
            }
        }
    )