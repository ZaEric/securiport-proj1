from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


# TODO: hold
# CURRENT_RUN_ID = "2026-06-29_1642_john_doe"
CURRENT_RUN_ID = "2026-07-09_1809_elizabeth_holmes"

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

class DebugRunInfo(BaseModel):
    run_id: str
    run_dir: str
    has_input: bool
    has_sources: bool
    has_chunks: bool
    has_evidence: bool
    has_aggregation: bool
    has_final_report: bool
    num_sources: int | None = None
    num_chunks: int | None = None
    num_evidence: int | None = None
    overall_sentiment: str | None = None


class DebugRunsResponse(BaseModel):
    runs: list[DebugRunInfo]


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
    max_urls: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of curated URLs to fetch.",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_name": "Elizabeth Holmes",
                "curated_urls_file": "data/curated/curated_urls.json",
                "max_urls": 2,
                "run_id": None,
            }
        }
    )


class FromSearchApiRequest(BaseModel):
    target_name: str
    search_provider: str = "tavily"
    search_query: str | None = Field(
        default=None,
        description="Optional search query. If omitted, one is generated from target_name.",
    )
    max_urls: int = Field(default=3, ge=1)
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_name": "John Doe",
                "search_provider": "tavily",
                "search_query": "\"John Doe\" news",
                "max_urls": 3,
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
    target_chunk_words: int = Field(default=3500, ge=1)
    max_chunk_words: int = Field(default=4000, ge=1)
    overlap_paragraphs: int = Field(default=1, ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "sources_path": None,
                "target_chunk_words": 3500,
                "max_chunk_words": 4000,
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
    model_name: str | None = Field(
        default=None,
        description="Optional evidence extraction model override. If omitted, uses OLLAMA_EVIDENCE_MODEL.",
    )

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
    mixed_evidence_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=0.5,
        description="Minimum opposing sentiment ratio needed to mark evidence as meaningfully mixed.",
    )
    baseline_tie_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="If positive and negative counts are within this difference ratio, baseline sentiment becomes neutral.",
    )
    minimum_evidence_count: int = Field(
        default=1,
        ge=0,
        description="Minimum number of evidence snippets required before evidence is considered sufficient.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "evidence_path": None,
                "mixed_evidence_threshold": 0.25,
                "baseline_tie_threshold": 0.10,
                "minimum_evidence_count": 1,
            }
        }
    )

class AggregationResponse(BaseModel):
    run_id: str
    target_name: str
    run_dir: str
    evidence_path: str
    aggregation_path: str
    baseline_sentiment: str
    num_sources: int
    num_evidence: int
    status: str


# -------------------------
# Final report request
# -------------------------

class ReportGenerateRequest(BaseModel):
    run_id: str
    model_name: str | None = Field(
        default=None,
        description="Optional final report model override. If omitted, uses OLLAMA_REPORT_MODEL.",
    )
    max_evidence_examples: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Maximum number of evidence examples included in final_report.json.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "run_id": CURRENT_RUN_ID,
                "model_name": None,
                "max_evidence_examples": 7,
            }
        }
    )

class ReportGenerateResponse(BaseModel):
    run_id: str
    target_name: str
    run_dir: str
    final_report_path: str
    overall_sentiment: str | None
    num_evidence_examples: int
    status: str

# -------------------------
# Full pipeline requests
# -------------------------

class SyntheticLLMRunRequest(BaseModel):
    synthetic_case_file: str
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )
    evidence_model_name: str | None = Field(
        default=None,
        description="Optional override for the evidence extraction LLM. If omitted, uses OLLAMA_EVIDENCE_MODEL.",
    )
    report_model_name: str | None = Field(
        default=None,
        description="Optional override for the final report LLM. If omitted, uses OLLAMA_REPORT_MODEL.",
    )
    max_evidence_examples: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Maximum number of evidence examples included in final_report.json.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "synthetic_case_file": "data/synthetic/passengers/synthetic_negative_001.json",
                "run_id": None,
                "evidence_model_name": None,
                "report_model_name": None,
                "max_evidence_examples": 7,
            }
        }
    )

class CuratedLLMRunRequest(BaseModel):
    target_name: str
    curated_urls_file: str
    max_urls: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of curated URLs to fetch.",
    )
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )
    evidence_model_name: str | None = Field(
        default=None,
        description="Optional override for the evidence extraction LLM. If omitted, uses OLLAMA_EVIDENCE_MODEL.",
    )
    report_model_name: str | None = Field(
        default=None,
        description="Optional override for the final report LLM. If omitted, uses OLLAMA_REPORT_MODEL.",
    )
    max_evidence_examples: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Maximum number of evidence examples included in final_report.json.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_name": "Elizabeth Holmes",
                "curated_urls_file": "data/curated/curated_urls.json",
                "max_urls": 3,
                "run_id": None,
                "evidence_model_name": None,
                "report_model_name": None,
                "max_evidence_examples": 7,
            }
        }
    )


class SearchLLMRunRequest(BaseModel):
    target_name: str
    search_provider: str = "tavily"
    search_query: str | None = Field(
        default=None,
        description="Optional search query. If omitted, one is generated from target_name.",
    )
    max_urls: int = Field(default=3, ge=1)
    run_id: str | None = Field(
        default=None,
        description="Optional run ID. If omitted, generated from datetime + target name.",
    )
    evidence_model_name: str | None = Field(
        default=None,
        description="Optional override for the evidence extraction LLM. If omitted, uses OLLAMA_EVIDENCE_MODEL.",
    )
    report_model_name: str | None = Field(
        default=None,
        description="Optional override for the final report LLM. If omitted, uses OLLAMA_REPORT_MODEL.",
    )
    max_evidence_examples: int = Field(
        default=7,
        ge=1,
        le=10,
        description="Maximum number of evidence examples included in final_report.json.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "target_name": "Elizabeth Holmes",
                "search_provider": "tavily",
                "search_query": "\"Elizabeth Holmes\" news",
                "max_urls": 3,
                "run_id": None,
                "evidence_model_name": None,
                "report_model_name": None,
                "max_evidence_examples": 7,
            }
        }
    )

    
class FullPipelineResponse(BaseModel):
    run_id: str
    target_name: str
    run_dir: str
    num_sources: int
    num_chunks: int
    num_evidence: int
    overall_sentiment: str | None
    status: str