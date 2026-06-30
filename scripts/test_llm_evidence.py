from pathlib import Path

from src.api.io import read_json, write_json
from src.nlp.pipeline import process_chunks_for_evidence_llm


run_id = "2026-06-29_1642_john_doe"

chunks_path = Path("data/runs") / run_id / "chunks.json"
evidence_path = Path("data/runs") / run_id / "evidence.json"

chunks_json = read_json(chunks_path)
evidence_json = process_chunks_for_evidence_llm(chunks_json)

write_json(evidence_path, evidence_json)

print(f"Wrote evidence to {evidence_path}")
print(f"Evidence count: {len(evidence_json['evidence'])}")