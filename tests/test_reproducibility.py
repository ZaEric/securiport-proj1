import json
from pathlib import Path
from typing import Any

import pytest


def test_llm_run_reproducibility(pytestconfig: pytest.Config) -> None:
    """
    Compares two completed run folders for near-identical LLM outputs.

    Expected usage:
        pytest tests/test_reproducibility.py --run-a data/runs/<first_run_id> --run-b data/runs/<second_run_id>

    This ignores run_id fields because run_id includes a timestamp.
    """
    run_a = get_required_run_path(pytestconfig, "--run-a")
    run_b = get_required_run_path(pytestconfig, "--run-b")

    assert run_a.exists(), f"Run folder does not exist: {run_a}"
    assert run_b.exists(), f"Run folder does not exist: {run_b}"

    evidence_a = load_json(run_a / "evidence.json")
    evidence_b = load_json(run_b / "evidence.json")

    final_report_a = load_json(run_a / "final_report.json")
    final_report_b = load_json(run_b / "final_report.json")

    assert evidence_a.get("target_name") == evidence_b.get("target_name")
    assert final_report_a.get("target_name") == final_report_b.get("target_name")

    assert get_num_evidence(evidence_a) == get_num_evidence(evidence_b)

    assert normalize_for_reproducibility(evidence_a) == normalize_for_reproducibility(evidence_b)

    assert final_report_a.get("evidence_examples") == final_report_b.get("evidence_examples")

    assert normalize_for_reproducibility(final_report_a) == normalize_for_reproducibility(final_report_b)


def get_required_run_path(pytestconfig: pytest.Config, option_name: str) -> Path:
    value = pytestconfig.getoption(option_name)

    if not value:
        pytest.skip(
            f"{option_name} is not set. "
            "Pass both --run-a and --run-b to compare two completed run folders."
        )

    return Path(value)


def load_json(path: Path) -> dict[str, Any]:
    assert path.exists(), f"Required artifact does not exist: {path}"

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, dict), f"Artifact must be a JSON object: {path}"

    return data


def get_num_evidence(evidence_json: dict[str, Any]) -> int:
    num_evidence = evidence_json.get("num_evidence")

    if isinstance(num_evidence, int):
        return num_evidence

    evidence = evidence_json.get("evidence")

    if isinstance(evidence, list):
        return len(evidence)

    raise AssertionError("Could not determine num_evidence from evidence.json")


def normalize_for_reproducibility(value: Any) -> Any:
    """
    Recursively removes run_id fields before comparison.
    """
    if isinstance(value, dict):
        return {
            key: normalize_for_reproducibility(item)
            for key, item in value.items()
            if key != "run_id"
        }

    if isinstance(value, list):
        return [normalize_for_reproducibility(item) for item in value]

    return value