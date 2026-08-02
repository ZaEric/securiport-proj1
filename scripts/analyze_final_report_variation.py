import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

## Run script with:
## python scripts/analyze_final_report_variation.py --reports-dir data\archive\example


TEXT_FIELDS = [
    "overall_sentiment",
    "aggregation_baseline_sentiment",
    "one_line_summary",
    "extended_summary",
    "justification",
]

LIST_FIELDS = [
    "key_findings",
    "limitations",
]

OUTPUT_FILENAME = "final_report_variation_analysis.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze variation across multiple final_report.json files."
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        help="Folder containing final_report JSON files to compare.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output markdown path. Defaults to <reports-dir>/final_report_variation_analysis.md.",
    )

    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    output_path = Path(args.output) if args.output else reports_dir / OUTPUT_FILENAME

    reports = load_reports(reports_dir)

    if not reports:
        raise SystemExit(f"No JSON reports found in: {reports_dir}")

    markdown = build_analysis_markdown(reports=reports, reports_dir=reports_dir)

    output_path.write_text(markdown, encoding="utf-8")

    print(f"Analyzed {len(reports)} reports.")
    print(f"Wrote analysis to: {output_path}")


def load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    for path in sorted(reports_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            continue

        data["_file_name"] = path.name
        reports.append(data)

    return reports


def build_analysis_markdown(
    reports: list[dict[str, Any]],
    reports_dir: Path,
) -> str:
    lines: list[str] = []

    lines.append("# Final Report Variation Analysis")
    lines.append("")
    lines.append(f"Reports directory: `{reports_dir}`")
    lines.append(f"Number of reports analyzed: **{len(reports)}**")
    lines.append("")

    lines.extend(build_basic_metadata_section(reports))
    lines.extend(build_text_field_variation_section(reports))
    lines.extend(build_list_field_variation_section(reports))
    lines.extend(build_evidence_example_section(reports))
    lines.extend(build_report_generation_issues_section(reports))

    return "\n".join(lines)


def build_basic_metadata_section(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    lines.append("## 1. Basic Metadata")
    lines.append("")

    for field in ["target_name", "overall_sentiment", "aggregation_baseline_sentiment"]:
        counter = Counter(normalize_value(report.get(field)) for report in reports)
        lines.append(f"### `{field}`")
        lines.append("")
        lines.extend(format_counter(counter))
        lines.append("")

    metadata_counter = Counter()

    for report in reports:
        metadata = report.get("llm_metadata", {})
        if isinstance(metadata, dict):
            key = json.dumps(metadata, sort_keys=True)
        else:
            key = normalize_value(metadata)

        metadata_counter[key] += 1

    lines.append("### `llm_metadata`")
    lines.append("")
    lines.extend(format_counter(metadata_counter))
    lines.append("")

    return lines


def build_text_field_variation_section(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    lines.append("## 2. Text Field Variation")
    lines.append("")

    for field in TEXT_FIELDS:
        counter = Counter(normalize_value(report.get(field)) for report in reports)
        unique_count = len(counter)

        lines.append(f"### `{field}`")
        lines.append("")
        lines.append(f"Unique variants: **{unique_count}** out of **{len(reports)}** reports")
        lines.append("")

        for index, (text, count) in enumerate(counter.most_common(), start=1):
            lines.append(f"#### Variant {index} — {count} report(s)")
            lines.append("")
            lines.append("> " + text.replace("\n", "\n> "))
            lines.append("")

    return lines


def build_list_field_variation_section(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    lines.append("## 3. List Field Variation")
    lines.append("")

    for field in LIST_FIELDS:
        exact_list_counter = Counter()
        item_counter = Counter()

        for report in reports:
            items = report.get(field, [])

            if not isinstance(items, list):
                items = []

            normalized_items = [
                item.strip()
                for item in items
                if isinstance(item, str) and item.strip()
            ]

            exact_list_counter[json.dumps(normalized_items, ensure_ascii=False)] += 1

            for item in normalized_items:
                item_counter[item] += 1

        lines.append(f"### `{field}`")
        lines.append("")
        lines.append(f"Unique full-list variants: **{len(exact_list_counter)}**")
        lines.append("")

        lines.append("#### Most common individual items")
        lines.append("")
        lines.extend(format_counter(item_counter))
        lines.append("")

        lines.append("#### Exact full-list variants")
        lines.append("")

        for index, (serialized_items, count) in enumerate(exact_list_counter.most_common(), start=1):
            items = json.loads(serialized_items)

            lines.append(f"Variant {index} — {count} report(s)")
            lines.append("")

            if items:
                for item in items:
                    lines.append(f"- {item}")
            else:
                lines.append("- None")

            lines.append("")

    return lines


def build_evidence_example_section(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    lines.append("## 4. Evidence Example Variation")
    lines.append("")

    evidence_count_counter = Counter()
    quote_counter = Counter()
    source_counter = Counter()
    quote_to_sources: dict[str, set[str]] = defaultdict(set)
    quote_to_sentiments: dict[str, set[str]] = defaultdict(set)
    quote_to_files: dict[str, list[str]] = defaultdict(list)

    report_to_quotes: dict[str, list[str]] = {}

    for report in reports:
        file_name = report.get("_file_name", "unknown")
        examples = report.get("evidence_examples", [])

        if not isinstance(examples, list):
            examples = []

        evidence_count_counter[len(examples)] += 1

        quotes_for_report: list[str] = []

        for example in examples:
            if not isinstance(example, dict):
                continue

            quote = normalize_value(example.get("quote"))
            source_id = normalize_value(example.get("source_id"))
            sentiment = normalize_value(example.get("sentiment"))

            if not quote:
                continue

            quote_counter[quote] += 1
            source_counter[source_id] += 1
            quote_to_sources[quote].add(source_id)
            quote_to_sentiments[quote].add(sentiment)
            quote_to_files[quote].append(file_name)
            quotes_for_report.append(quote)

        report_to_quotes[file_name] = quotes_for_report

    lines.append("### Number of evidence examples selected per report")
    lines.append("")
    lines.extend(format_counter(evidence_count_counter))
    lines.append("")

    lines.append("### Most commonly selected evidence quotes")
    lines.append("")

    for index, (quote, count) in enumerate(quote_counter.most_common(), start=1):
        sources = ", ".join(sorted(quote_to_sources[quote]))
        sentiments = ", ".join(sorted(quote_to_sentiments[quote]))

        lines.append(f"#### Evidence quote {index} — selected in {count}/{len(reports)} report(s)")
        lines.append("")
        lines.append(f"Source(s): `{sources}`")
        lines.append("")
        lines.append(f"Sentiment(s): `{sentiments}`")
        lines.append("")
        lines.append("> " + quote.replace("\n", "\n> "))
        lines.append("")

    lines.append("### Evidence quotes by report")
    lines.append("")

    for file_name, quotes in sorted(report_to_quotes.items()):
        lines.append(f"#### `{file_name}`")
        lines.append("")

        if not quotes:
            lines.append("- No evidence examples selected.")
        else:
            for quote in quotes:
                lines.append(f"- {quote}")

        lines.append("")

    return lines


def build_report_generation_issues_section(reports: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []

    lines.append("## 5. Report Generation Issues")
    lines.append("")

    issues_counter = Counter()

    for report in reports:
        issues = report.get("report_generation_issues", [])

        if not isinstance(issues, list):
            issues_counter["report_generation_issues was not a list"] += 1
            continue

        if not issues:
            issues_counter["No issues"] += 1
            continue

        for issue in issues:
            if isinstance(issue, str):
                issues_counter[issue] += 1
            else:
                issues_counter[json.dumps(issue, sort_keys=True, ensure_ascii=False)] += 1

    lines.extend(format_counter(issues_counter))
    lines.append("")

    return lines


def format_counter(counter: Counter[Any]) -> list[str]:
    lines: list[str] = []

    if not counter:
        return ["- None"]

    total = sum(counter.values())

    for value, count in counter.most_common():
        percentage = count / total * 100 if total else 0
        lines.append(f"- **{count}** / {total} ({percentage:.1f}%): `{value}`")

    return lines


def normalize_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    return str(value)


if __name__ == "__main__":
    main()