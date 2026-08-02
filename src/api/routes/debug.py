from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.api.io import path_to_api_str, read_json
from src.api.schemas import DebugRunInfo, DebugRunsResponse

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/runs", response_model=DebugRunsResponse)
def list_runs() -> DebugRunsResponse:
    runs_dir = Path("data/runs")

    if not runs_dir.exists():
        return DebugRunsResponse(runs=[])

    run_infos = [
        build_debug_run_info(run_dir)
        for run_dir in sorted(runs_dir.iterdir(), reverse=True)
        if run_dir.is_dir()
    ]

    return DebugRunsResponse(runs=run_infos)


def build_debug_run_info(run_dir: Path) -> DebugRunInfo:
    input_path = run_dir / "input.json"
    sources_path = run_dir / "sources.json"
    chunks_path = run_dir / "chunks.json"
    evidence_path = run_dir / "evidence.json"
    aggregation_path = run_dir / "aggregation.json"
    final_report_path = run_dir / "final_report.json"

    sources_json = safe_read_json(sources_path)
    chunks_json = safe_read_json(chunks_path)
    evidence_json = safe_read_json(evidence_path)
    final_report_json = safe_read_json(final_report_path)

    return DebugRunInfo(
        run_id=run_dir.name,
        run_dir=path_to_api_str(run_dir),
        has_input=input_path.exists(),
        has_sources=sources_path.exists(),
        has_chunks=chunks_path.exists(),
        has_evidence=evidence_path.exists(),
        has_aggregation=aggregation_path.exists(),
        has_final_report=final_report_path.exists(),
        num_sources=count_items(sources_json, "sources"),
        num_chunks=count_items(chunks_json, "chunks"),
        num_evidence=get_num_evidence(evidence_json),
        overall_sentiment=get_overall_sentiment(final_report_json),
    )


def safe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        return read_json(path)
    except Exception:
        return None


def count_items(data: dict[str, Any] | None, key: str) -> int | None:
    if data is None:
        return None

    items = data.get(key)

    if isinstance(items, list):
        return len(items)

    return None


def get_num_evidence(evidence_json: dict[str, Any] | None) -> int | None:
    if evidence_json is None:
        return None

    num_evidence = evidence_json.get("num_evidence")

    if isinstance(num_evidence, int):
        return num_evidence

    evidence = evidence_json.get("evidence")

    if isinstance(evidence, list):
        return len(evidence)

    return None


def get_overall_sentiment(final_report_json: dict[str, Any] | None) -> str | None:
    if final_report_json is None:
        return None

    overall_sentiment = final_report_json.get("overall_sentiment")

    if isinstance(overall_sentiment, str):
        return overall_sentiment

    return None