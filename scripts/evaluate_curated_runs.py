import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CURATED_URLS_FILE = Path("data/curated/curated_urls.json")
DEFAULT_RUNS_DIR = Path("data/runs")
DEFAULT_OUTPUT_PATH = Path("data/evaluation/curated_runs_summary.json")


def main() -> None:
    args = parse_args()

    expected_by_person = load_expected_sentiments(args.curated_urls_file)
    run_records = load_curated_run_records(args.runs_dir)
    latest_runs = select_latest_run_per_person(run_records)

    rows = []

    for person, expected_sentiment in expected_by_person.items():
        run_record = latest_runs.get(person.lower())
        row = build_evaluation_row(
            person=person,
            expected_sentiment=expected_sentiment,
            run_record=run_record,
        )
        rows.append(row)

    print_table(rows)
    print_summary(rows)

    if args.write_json:
        write_json_summary(args.output_path, rows)
        print(f"\nWrote {args.output_path.as_posix()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate curated runs against expected sentiment labels.",
    )
    parser.add_argument(
        "--curated-urls-file",
        type=Path,
        default=DEFAULT_CURATED_URLS_FILE,
        help="Path to data/curated/curated_urls.json.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing run artifacts.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write evaluation summary JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for --write-json output.",
    )
    return parser.parse_args()


def load_expected_sentiments(path: Path) -> dict[str, str | None]:
    data = read_json(path)

    if not isinstance(data, list):
        raise ValueError("curated URLs file must contain a list of person records.")

    expected_by_person: dict[str, str | None] = {}

    for item in data:
        if not isinstance(item, dict):
            continue

        person = item.get("person")
        expected_sentiment = item.get("expected_sentiment")

        if not isinstance(person, str) or not person.strip():
            continue

        expected_by_person[person.strip()] = (
            expected_sentiment if isinstance(expected_sentiment, str) else None
        )

    return expected_by_person


def load_curated_run_records(runs_dir: Path) -> list[dict[str, Any]]:
    records = []

    if not runs_dir.exists():
        return records

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue

        input_path = run_dir / "input.json"
        sources_path = run_dir / "sources.json"
        evidence_path = run_dir / "evidence.json"
        aggregation_path = run_dir / "aggregation.json"

        if not input_path.exists() or not aggregation_path.exists():
            continue

        input_json = read_json(input_path)

        if input_json.get("input_mode") != "curated_urls":
            continue

        aggregation_json = read_json(aggregation_path)
        sources_json = read_json(sources_path) if sources_path.exists() else {}
        evidence_json = read_json(evidence_path) if evidence_path.exists() else {}

        records.append(
            {
                "run_dir": run_dir,
                "input_path": input_path,
                "sources_path": sources_path,
                "evidence_path": evidence_path,
                "aggregation_path": aggregation_path,
                "input_json": input_json,
                "sources_json": sources_json,
                "evidence_json": evidence_json,
                "aggregation_json": aggregation_json,
                "modified_at": input_path.stat().st_mtime,
            }
        )

    return records


def select_latest_run_per_person(
    run_records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest_runs: dict[str, dict[str, Any]] = {}

    for record in run_records:
        target_name = record["input_json"].get("target_name")

        if not isinstance(target_name, str):
            continue

        key = target_name.strip().lower()
        existing = latest_runs.get(key)

        if existing is None or record["modified_at"] > existing["modified_at"]:
            latest_runs[key] = record

    return latest_runs


def build_evaluation_row(
    *,
    person: str,
    expected_sentiment: str | None,
    run_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if run_record is None:
        return {
            "target_name": person,
            "run_id": None,
            "expected_sentiment": expected_sentiment,
            "baseline_sentiment": None,
            "match": False,
            "source_count": 0,
            "evidence_count": 0,
            "mixed_evidence": None,
            "insufficient_evidence": None,
            "notes": "missing curated run with aggregation.json",
        }

    input_json = run_record["input_json"]
    sources_json = run_record["sources_json"]
    evidence_json = run_record["evidence_json"]
    aggregation_json = run_record["aggregation_json"]
    overall_result = aggregation_json.get("overall_result", {})

    baseline_sentiment = overall_result.get("baseline_sentiment")
    match = (
        expected_sentiment is not None
        and baseline_sentiment is not None
        and expected_sentiment == baseline_sentiment
    )

    return {
        "target_name": input_json.get("target_name"),
        "run_id": input_json.get("run_id"),
        "expected_sentiment": expected_sentiment,
        "baseline_sentiment": baseline_sentiment,
        "match": match,
        "source_count": len(sources_json.get("sources", [])),
        "evidence_count": evidence_json.get(
            "num_evidence",
            len(evidence_json.get("evidence", [])),
        ),
        "mixed_evidence": overall_result.get("mixed_evidence"),
        "insufficient_evidence": overall_result.get("insufficient_evidence"),
        "sentiment_counts": overall_result.get("sentiment_counts", {}),
        "notes": build_notes(expected_sentiment, baseline_sentiment, match),
    }


def build_notes(
    expected_sentiment: str | None,
    baseline_sentiment: str | None,
    match: bool,
) -> str:
    if expected_sentiment is None:
        return "missing expected sentiment"

    if baseline_sentiment is None:
        return "missing aggregation baseline sentiment"

    if match:
        return ""

    return "expected sentiment does not match aggregation baseline"


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Target",
        "Expected",
        "Baseline",
        "Match",
        "Evidence",
        "Sources",
        "Mixed",
        "Insufficient",
    ]
    table_rows = [
        [
            str(row["target_name"]),
            value_or_dash(row["expected_sentiment"]),
            value_or_dash(row["baseline_sentiment"]),
            "yes" if row["match"] else "no",
            str(row["evidence_count"]),
            str(row["source_count"]),
            bool_to_text(row["mixed_evidence"]),
            bool_to_text(row["insufficient_evidence"]),
        ]
        for row in rows
    ]

    widths = calculate_column_widths(headers, table_rows)
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))

    for table_row in table_rows:
        print(format_row(table_row, widths))


def print_summary(rows: list[dict[str, Any]]) -> None:
    evaluable_rows = [
        row
        for row in rows
        if row["expected_sentiment"] is not None
        and row["baseline_sentiment"] is not None
    ]
    matched_rows = [row for row in evaluable_rows if row["match"]]
    mismatched_rows = [row for row in evaluable_rows if not row["match"]]

    print("\nSummary")
    print(f"  evaluable_runs: {len(evaluable_rows)}")
    print(f"  matched: {len(matched_rows)}")
    print(f"  mismatched: {len(mismatched_rows)}")

    if mismatched_rows:
        print("  mismatches:")
        for row in mismatched_rows:
            print(
                "    "
                f"{row['target_name']}: "
                f"expected={row['expected_sentiment']} "
                f"baseline={row['baseline_sentiment']}"
            )


def calculate_column_widths(
    headers: list[str],
    rows: list[list[str]],
) -> list[int]:
    widths = [len(header) for header in headers]

    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    return widths


def format_row(values: list[str], widths: list[int]) -> str:
    padded_values = [
        value.ljust(widths[index])
        for index, value in enumerate(values)
    ]
    return "  ".join(padded_values)


def value_or_dash(value: Any) -> str:
    return "-" if value is None else str(value)


def bool_to_text(value: Any) -> str:
    if value is True:
        return "yes"

    if value is False:
        return "no"

    return "-"


def write_json_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "num_cases": len(rows),
        "num_matched": sum(1 for row in rows if row["match"]),
        "num_mismatched": sum(
            1
            for row in rows
            if row["expected_sentiment"] is not None
            and row["baseline_sentiment"] is not None
            and not row["match"]
        ),
        "results": rows,
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()
