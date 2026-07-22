from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.io import path_to_api_str, read_json, write_json
from src.api.schemas import ReportGenerateRequest, ReportGenerateResponse
from src.nlp.models import EvidenceExtractionConfig
from src.nlp.pipeline import process_final_report_llm


router = APIRouter(prefix="/report", tags=["report"])


@router.post("/generate", response_model=ReportGenerateResponse)
def generate_report(request: ReportGenerateRequest) -> ReportGenerateResponse:
    run_dir = Path("data/runs") / request.run_id

    # Standard artifact paths for this run.
    sources_path = run_dir / "sources.json"
    evidence_path = run_dir / "evidence.json"
    aggregation_path = run_dir / "aggregation.json"
    final_report_path = run_dir / "final_report.json"

    ensure_file_exists(sources_path, "Sources")
    ensure_file_exists(evidence_path, "Evidence")
    ensure_file_exists(aggregation_path, "Aggregation")

    try:
        sources_json = read_json(sources_path)
        evidence_json = read_json(evidence_path)
        aggregation_json = read_json(aggregation_path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read report input files: {exc}",
        ) from exc

    validate_run_id(request.run_id, sources_json, "sources.json")
    validate_run_id(request.run_id, evidence_json, "evidence.json")
    validate_run_id(request.run_id, aggregation_json, "aggregation.json")

    # Optional model override. If omitted, EvidenceExtractionConfig uses the default model.
    config = EvidenceExtractionConfig(
        model_name=request.model_name if request.model_name else EvidenceExtractionConfig().model_name,
    )

    try:
        final_report_json = process_final_report_llm(
            sources_json=sources_json,
            evidence_json=evidence_json,
            aggregation_json=aggregation_json,
            config=config,
            max_evidence_examples=request.max_evidence_examples,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Final report generation failed: {exc}",
        ) from exc

    write_json(final_report_path, final_report_json)

    return ReportGenerateResponse(
        run_id=final_report_json["run_id"],
        target_name=final_report_json["target_name"],
        run_dir=path_to_api_str(run_dir),
        final_report_path=path_to_api_str(final_report_path),
        overall_sentiment=final_report_json["overall_sentiment"],
        num_evidence_examples=len(final_report_json["evidence_examples"]),
        status="created",
    )


def ensure_file_exists(path: Path, label: str) -> None:
    """
    Raises a clear 404 if a required report input file is missing.
    """
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{label} file not found: {path}",
        )


def validate_run_id(run_id: str, artifact_json: dict, artifact_name: str) -> None:
    """
    Prevents accidentally mixing artifacts from different runs.
    """
    artifact_run_id = artifact_json.get("run_id")

    if artifact_run_id != run_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Run ID mismatch in {artifact_name}: "
                f"request run_id={run_id}, artifact run_id={artifact_run_id}"
            ),
        )