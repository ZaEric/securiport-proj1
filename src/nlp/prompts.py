import json
from typing import Any
def build_evidence_extraction_prompt(target_name: str, chunk_text: str) -> str:
    return f"""
You are extracting evidence about a target person from a text chunk.

Target person: {target_name}

Task:
Extract the most relevant distinct evidence spans about the target person from the provided text.

For this project, negative sentiment means evidence that may require closer human review. Negative evidence does not always mean the target person did something wrong.

Rules:
- Use only the provided text.
- For the "quote" field, behave like a copy-paste tool.
- Each quote must be copied exactly from the text.
- Each quote must be one continuous substring from the text.
- Preserve exact casing, punctuation, wording, and verb tense.
- Do not paraphrase, summarize, shorten, or rewrite evidence.
- Do not add attribution, names, punctuation, or context that is not part of the selected text span.
- Do not use ellipses (...) or (…) to skip words inside a quote.
- Do not stitch together separate parts of the text into one quote.
- Do not combine the beginning of one sentence with the end of another sentence.
- Prefer complete sentences as evidence quotes.
- Do not return sentence fragments unless the fragment is self-contained and clearly meaningful.
- Ignore evidence about people other than the target person.
- Each quote must have its own sentiment label.
- Sentiment must be one of: positive, neutral, negative.
- Do not include confidence scores.
- Do not extract every relevant sentence.
- Focus on the strongest evidence that would help justify a sentiment or review-flag judgment.
- Avoid repetitive, overlapping, or minor detail evidence.
- Return multiple quotes only when they capture distinct important facts or distinct sentiment.
- If there is no relevant evidence, return an empty evidence list.
- Return JSON only.

Sentiment guidance:
- Label evidence as negative if it may require closer officer or analyst review.
- Label misconduct, criminal activity, fraud, legal penalties, audit findings, sanctions, removal from role, failure, criticism, security concerns, or reputational harm as negative.
- Label evidence that the target person was a victim of crime, exploitation, trafficking, coercion, violence, threats, or other harmful circumstances as negative, because it may require further review or protection.
- Do not imply wrongdoing when negative evidence only indicates that the target person may be a victim or affected party.
- Label praise, awards, achievements, community support, or clearly favorable actions as positive.
- Label background, procedural details, explanations, denials, or role descriptions as neutral unless they clearly support positive, negative, or review-relevant sentiment.

Invalid quote examples:
- Do not change casing: "over the next year" if the text says "Over the next year"
- Do not change wording: "fool" if the text says "fooled"
- Do not change quotation marks: 'example' if the text says "example"
- Do not stitch text together: "In 2015, Forbes revised..." if "In 2015" and "Forbes revised" appear in separate sentences
- Do not use ellipses: "John Doe ... was convicted"

Required JSON format:
{{
  "evidence": [
    {{
      "quote": "one exact continuous text span copied character-for-character from the text",
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

Important interpretation:
- In this system, negative sentiment means a review flag.
- Negative sentiment does not always mean the target person committed wrongdoing.
- Negative evidence may indicate misconduct, legal concerns, reputational harm, security concerns, victimization, exploitation, threats, or other circumstances that may require closer human review.
- If negative evidence only shows that the target person may be a victim or affected party, do not imply that the target person committed wrongdoing.

Important rules:
- Use only the supplied evidence.
- Do not invent facts.
- Do not use outside knowledge.
- The final overall_sentiment must be one of: positive, neutral, negative.
- Do not use "mixed" as the final overall_sentiment.
- The final overall_sentiment should follow the supplied evidence statistics and review-flag logic.
- If negative evidence is present, explain why the report is negative as a review flag.
- Evidence statistics are provided to summarize the deterministic aggregation result.
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