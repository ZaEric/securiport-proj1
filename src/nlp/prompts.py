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