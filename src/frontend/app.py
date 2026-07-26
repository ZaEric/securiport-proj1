from pathlib import Path
from typing import Any

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
RUNS_DIR = Path("data/runs")

ARTIFACT_FILES = [
    "input.json",
    "sources.json",
    "chunks.json",
    "evidence.json",
    "aggregation.json",
    "final_report.json",
]


st.set_page_config(
    page_title="Securiport Sentiment Dashboard",
    page_icon="🧾",
    layout="wide",
)


def main() -> None:
    st.title("Securiport Sentiment Analysis Dashboard")

    st.caption(
        "Simple frontend for running the pipeline and viewing generated JSON artifacts."
    )

    with st.sidebar:
        st.header("Controls")

        api_base_url = st.text_input(
            "FastAPI base URL",
            value=API_BASE_URL,
        )

        page = st.radio(
            "Mode",
            options=[
                "Run Full Pipeline",
                "View Existing Run",
            ],
        )

    if page == "Run Full Pipeline":
        render_run_pipeline_page(api_base_url)
    else:
        render_existing_run_page()


def render_run_pipeline_page(api_base_url: str) -> None:
    st.header("Run Full Pipeline")

    pipeline_mode = st.selectbox(
        "Pipeline input mode",
        options=[
            "Synthetic",
            "Curated URLs",
            "Search API",
        ],
    )

    if pipeline_mode == "Synthetic":
        render_synthetic_pipeline_form(api_base_url)
    elif pipeline_mode == "Curated URLs":
        render_curated_pipeline_form(api_base_url)
    else:
        render_search_pipeline_form(api_base_url)


def render_synthetic_pipeline_form(api_base_url: str) -> None:
    st.subheader("Synthetic Full Run")

    synthetic_case_file = st.text_input(
        "Synthetic case file",
        value="data/synthetic/passengers/synthetic_negative_001.json",
    )
    max_evidence_examples = st.slider(
        "Max evidence examples",
        min_value=1,
        max_value=10,
        value=7,
    )

    if st.button("Run Synthetic Pipeline", type="primary"):
        payload = {
            "synthetic_case_file": synthetic_case_file,
            "run_id": None,
            "model_name": None,
            "max_evidence_examples": max_evidence_examples,
        }

        call_pipeline_endpoint(
            api_base_url=api_base_url,
            endpoint="/pipeline/run-synthetic-llm",
            payload=payload,
        )


def render_curated_pipeline_form(api_base_url: str) -> None:
    st.subheader("Curated URLs Full Run")

    target_name = st.text_input("Target name", value="Elizabeth Holmes")
    curated_urls_file = st.text_input(
        "Curated URLs file",
        value="data/curated/curated_urls.json",
    )
    max_urls = st.number_input(
        "Max URLs",
        min_value=1,
        value=3,
    )
    max_evidence_examples = st.slider(
        "Max evidence examples",
        min_value=1,
        max_value=10,
        value=7,
    )

    if st.button("Run Curated Pipeline", type="primary"):
        payload = {
            "target_name": target_name,
            "curated_urls_file": curated_urls_file,
            "max_urls": max_urls,
            "run_id": None,
            "model_name": None,
            "max_evidence_examples": max_evidence_examples,
        }

        call_pipeline_endpoint(
            api_base_url=api_base_url,
            endpoint="/pipeline/run-curated-llm",
            payload=payload,
        )


def render_search_pipeline_form(api_base_url: str) -> None:
    st.subheader("Search API Full Run")

    target_name = st.text_input("Target name", value="Elizabeth Holmes")
    search_provider = st.text_input("Search provider", value="tavily")
    max_urls = st.number_input(
        "Max URLs",
        min_value=1,
        value=3,
    )
    max_evidence_examples = st.slider(
        "Max evidence examples",
        min_value=1,
        max_value=10,
        value=7,
    )

    if st.button("Run Search Pipeline", type="primary"):
        payload = {
            "target_name": target_name,
            "search_provider": search_provider,
            "search_query": None,
            "max_urls": max_urls,
            "run_id": None,
            "model_name": None,
            "max_evidence_examples": max_evidence_examples,
        }

        call_pipeline_endpoint(
            api_base_url=api_base_url,
            endpoint="/pipeline/run-search-llm",
            payload=payload,
        )


def call_pipeline_endpoint(
    *,
    api_base_url: str,
    endpoint: str,
    payload: dict[str, Any],
) -> None:
    url = f"{api_base_url.rstrip('/')}{endpoint}"

    with st.spinner("Running pipeline... this may take a while."):
        try:
            response = requests.post(url, json=payload, timeout=300)
        except requests.RequestException as exc:
            st.error(f"Failed to call API: {exc}")
            return

    if response.ok:
        result = response.json()
        st.success("Pipeline completed.")
        st.json(result)

        run_id = result.get("run_id")
        if isinstance(run_id, str):
            st.divider()
            render_run_artifacts(run_id)
    else:
        st.error(f"API returned {response.status_code}")
        try:
            st.json(response.json())
        except Exception:
            st.text(response.text)


def render_existing_run_page() -> None:
    st.header("View Existing Run")

    run_ids = list_run_ids()

    if not run_ids:
        st.warning("No runs found under data/runs.")
        return

    st.write(f"Found **{len(run_ids)}** run(s).")

    run_table = [
        {
            "run_id": run_id,
            "has_input": (RUNS_DIR / run_id / "input.json").exists(),
            "has_sources": (RUNS_DIR / run_id / "sources.json").exists(),
            "has_chunks": (RUNS_DIR / run_id / "chunks.json").exists(),
            "has_evidence": (RUNS_DIR / run_id / "evidence.json").exists(),
            "has_aggregation": (RUNS_DIR / run_id / "aggregation.json").exists(),
            "has_final_report": (RUNS_DIR / run_id / "final_report.json").exists(),
        }
        for run_id in run_ids
    ]

    st.dataframe(run_table, use_container_width=True)

    selected_run_id = st.selectbox(
        "Select run_id to inspect",
        options=run_ids,
    )

    render_run_artifacts(selected_run_id)


def list_run_ids() -> list[str]:
    runs_dir = RUNS_DIR.resolve()

    if not runs_dir.exists():
        return []

    return sorted(
        [path.name for path in runs_dir.iterdir() if path.is_dir()],
        reverse=True,
    )


def render_run_artifacts(run_id: str) -> None:
    run_dir = RUNS_DIR / run_id

    st.subheader(f"Run: `{run_id}`")

    if not run_dir.exists():
        st.error(f"Run directory not found: {run_dir}")
        return

    render_run_summary(run_dir)

    st.divider()

    tabs = st.tabs(ARTIFACT_FILES)

    for tab, artifact_name in zip(tabs, ARTIFACT_FILES):
        with tab:
            render_artifact(run_dir, artifact_name)

    render_extra_json_files(run_dir)


def render_run_summary(run_dir: Path) -> None:
    sources_json = read_json_if_exists(run_dir / "sources.json")
    chunks_json = read_json_if_exists(run_dir / "chunks.json")
    evidence_json = read_json_if_exists(run_dir / "evidence.json")
    final_report_json = read_json_if_exists(run_dir / "final_report.json")

    num_sources = count_list_field(sources_json, "sources")
    num_chunks = count_list_field(chunks_json, "chunks")
    num_evidence = get_num_evidence(evidence_json)
    overall_sentiment = get_string_field(final_report_json, "overall_sentiment")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Sources", display_metric_value(num_sources))
    col2.metric("Chunks", display_metric_value(num_chunks))
    col3.metric("Evidence", display_metric_value(num_evidence))
    col4.metric("Overall Sentiment", overall_sentiment or "N/A")

    if final_report_json:
        render_final_report(final_report_json)


def render_artifact(run_dir: Path, artifact_name: str) -> None:
    artifact_path = run_dir / artifact_name

    if not artifact_path.exists():
        st.info(f"`{artifact_name}` not found.")
        return

    artifact_json = read_json_if_exists(artifact_path)

    if artifact_json is None:
        st.error(f"Failed to read `{artifact_name}`.")
        return

    st.caption(str(artifact_path))
    st.json(artifact_json)

def render_final_report(final_report_json: dict[str, Any]) -> None:
    st.markdown("### Final Report")

    st.markdown("#### Summary")
    st.write(f"**Target:** {final_report_json.get('target_name', 'N/A')}")
    st.write(f"**Overall sentiment:** {final_report_json.get('overall_sentiment', 'N/A')}")
    st.write(f"**Aggregation baseline sentiment:** {final_report_json.get('aggregation_baseline_sentiment', 'N/A')}")
    st.write(f"**One-line summary:** {final_report_json.get('one_line_summary', 'N/A')}")

    extended_summary = final_report_json.get("extended_summary")
    if isinstance(extended_summary, str) and extended_summary:
        st.markdown("#### Extended Summary")
        st.write(extended_summary)

    key_findings = final_report_json.get("key_findings", [])
    if isinstance(key_findings, list) and key_findings:
        st.markdown("#### Key Findings")
        for finding in key_findings:
            st.markdown(f"- {finding}")

    justification = final_report_json.get("justification")
    if isinstance(justification, str) and justification:
        st.markdown("#### Justification")
        st.write(justification)

    evidence_examples = final_report_json.get("evidence_examples", [])
    if isinstance(evidence_examples, list) and evidence_examples:
        st.markdown("#### Evidence Examples")
        for index, evidence in enumerate(evidence_examples, start=1):
            if not isinstance(evidence, dict):
                continue

            with st.expander(f"Evidence {index}: {evidence.get('sentiment', 'N/A')} from {evidence.get('source_id', 'N/A')}"):
                st.write(f"**Source ID:** {evidence.get('source_id', 'N/A')}")
                st.write(f"**Sentiment:** {evidence.get('sentiment', 'N/A')}")
                st.write(f"**Quote:** {evidence.get('quote', 'N/A')}")
                st.write(f"**Why selected:** {evidence.get('why_selected', 'N/A')}")

    sources = final_report_json.get("sources", [])
    if isinstance(sources, list) and sources:
        st.markdown("#### Sources")
        st.dataframe(sources, use_container_width=True)

    limitations = final_report_json.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        st.markdown("#### Limitations")
        for limitation in limitations:
            st.markdown(f"- {limitation}")

def render_extra_json_files(run_dir: Path) -> None:
    standard_files = set(ARTIFACT_FILES)
    extra_json_files = sorted(
        [
            path
            for path in run_dir.glob("*.json")
            if path.name not in standard_files
        ]
    )

    if not extra_json_files:
        return

    st.divider()
    st.subheader("Additional JSON Files")

    selected_file = st.selectbox(
        "Select additional JSON file",
        options=[path.name for path in extra_json_files],
    )

    selected_path = run_dir / selected_file
    selected_json = read_json_if_exists(selected_path)

    if selected_json is None:
        st.error(f"Failed to read `{selected_file}`.")
        return

    st.caption(str(selected_path))
    st.json(selected_json)


def read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None

    try:
        import json

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def count_list_field(data: Any, key: str) -> int | None:
    if not isinstance(data, dict):
        return None

    value = data.get(key)

    if isinstance(value, list):
        return len(value)

    return None


def get_num_evidence(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None

    num_evidence = data.get("num_evidence")

    if isinstance(num_evidence, int):
        return num_evidence

    evidence = data.get("evidence")

    if isinstance(evidence, list):
        return len(evidence)

    return None


def get_string_field(data: Any, key: str) -> str | None:
    if not isinstance(data, dict):
        return None

    value = data.get(key)

    if isinstance(value, str):
        return value

    return None


def display_metric_value(value: int | None) -> str:
    if value is None:
        return "N/A"

    return str(value)


if __name__ == "__main__":
    main()