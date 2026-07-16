import json
from typing import Any
def build_evidence_extraction_prompt(target_name: str, chunk_text: str) -> str:
    return f"""
You are extracting evidence about a target person from a text chunk.

Target person: {target_name}

Task:
Extract the most relevant distinct evidence quotes about the target person from the provided text.

Rules:
- Use only the provided text.
- Extract exact quotes only.
- Do not paraphrase.
- Ignore evidence about people other than the target person.
- Each quote must be copied exactly from the text.
- Each quote must have its own sentiment label.
- Sentiment must be one of: positive, neutral, negative.
- Do not include confidence scores.
- Do not extract every relevant sentence.
- Focus on the strongest evidence that would help justify a sentiment judgment.
- Avoid repetitive, overlapping, or minor detail quotes.
- Return multiple quotes only when they capture distinct important facts or distinct sentiment.
- If there is no relevant evidence, return an empty evidence list.
- Return JSON only.

Sentiment guidance:
- Label misconduct, legal penalties, audit findings, sanctions, removal from role, failure, criticism, or reputational harm as negative.
- Label praise, awards, achievements, community support, or clearly favorable actions as positive.
- Label background, procedural details, explanations, denials, or role descriptions as neutral unless they clearly support positive or negative sentiment.

Required JSON format:
{{
  "evidence": [
    {{
      "quote": "exact quote from the text",
      "sentiment": "positive | neutral | negative"
    }}
  ]
}}

Text chunk:
\"\"\"
{chunk_text}
\"\"\"
""".strip()


def build_final_report_prompt(
    target_name: str,
    sources_summary: list[dict[str, Any]],
    evidence_json: dict[str, Any],
    aggregation_summary: dict[str, Any],
    max_evidence_examples: int = 7,
) -> str:
    return f"""
You are generating a grounded analytical report about a target person.

Target person: {target_name}

Task:
Use the supplied source metadata, extracted evidence, and evidence statistics to generate a final structured report.

Important rules:
- Use only the supplied evidence.
- Do not invent facts.
- Do not use outside knowledge.
- The final overall_sentiment must be one of: positive, neutral, negative.
- Do not use "mixed" as the final overall_sentiment.
- Base the final sentiment primarily on the evidence quotes, not only on raw counts.
- Evidence statistics are provided only as summary context.
- Do not introduce stronger legal, safety, criminal, or misconduct terms than the evidence supports.
- Use the same terminology as the evidence where possible.
- Include at most {max_evidence_examples} evidence examples in the final output.
- Evidence examples must come from the supplied evidence list.
- Prefer the strongest, clearest, least repetitive evidence examples.
- Evidence examples in the final output should include source_id, quote, sentiment, and why_selected.
- Do not include evidence_id in the final output.
- If evidence is mixed or insufficient, explain that in the justification or limitations.
- Return JSON only.

Required JSON format:
{{
  "overall_sentiment": "positive | neutral | negative",
  "overall_confidence": null,
  "one_line_summary": "one sentence summary",
  "extended_summary": "short paragraph summary",
  "key_findings": [
    "finding 1",
    "finding 2",
    "finding 3"
  ],
  "justification": "explanation of why the overall sentiment was chosen",
  "evidence_examples": [
    {{
      "source_id": "src_001",
      "quote": "exact quote from the supplied evidence",
      "sentiment": "negative",
      "why_selected": "brief reason this quote supports the report"
    }}
  ],
  "sources": [
    {{
      "source_id": "src_001",
      "url": "source URL",
      "title": "source title"
    }}
  ],
  "limitations": [
    "limitation 1",
    "limitation 2"
  ]
}}

Sources Summary:
{json.dumps(sources_summary, indent=2)}

Evidence JSON:
{json.dumps(evidence_json, indent=2)}

Evidence Statistics:
{json.dumps(aggregation_summary, indent=2)}
""".strip()