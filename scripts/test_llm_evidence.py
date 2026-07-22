import argparse
from pathlib import Path

from src.api.io import read_json, write_json
from src.nlp.pipeline import process_chunks_for_evidence_llm


def run_for(run_id: str) -> None:
    chunks_path = Path("data/runs") / run_id / "chunks.json"
    evidence_path = Path("data/runs") / run_id / "evidence.json"

    chunks_json = read_json(chunks_path)
    evidence_json = process_chunks_for_evidence_llm(chunks_json)

    write_json(evidence_path, evidence_json)

    print(f"Wrote evidence to {evidence_path}")
    print(f"Evidence count: {len(evidence_json['evidence'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract LLM evidence for one or more runs under data/runs/."
    )
    parser.add_argument(
        "run_ids",
        nargs="+",
        help="Run id(s) under data/runs/ to process, e.g. 2026-06-29_1642_john_doe",
    )
    args = parser.parse_args()

    for run_id in args.run_ids:
        run_for(run_id)