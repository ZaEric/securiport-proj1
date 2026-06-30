from fastapi import APIRouter

from src.api.schemas import ReportGenerateRequest, StatusResponse


router = APIRouter(prefix="/report", tags=["report"])


@router.post("/generate", response_model=StatusResponse)
def generate_report(request: ReportGenerateRequest) -> StatusResponse:
    return StatusResponse(
        status="not_implemented",
        message="Endpoint contract reserved. Future behavior: sources.json + evidence.json + aggregation.json -> final_report.json.",
    )