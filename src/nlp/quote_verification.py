import re
import unicodedata


PUNCTUATION_REPLACEMENTS = {
    # Single quotes / apostrophes
    "\u2018": "'",
    "\u2019": "'",
    "\u201A": "'",
    "\u201B": "'",
    "\u2032": "'",

    # Double quotes
    "\u201C": '"',
    "\u201D": '"',
    "\u201E": '"',
    "\u201F": '"',
    "\u2033": '"',

    # Dashes / hyphens
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",

    # Spaces
    "\u00A0": " ",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_quote_for_matching(text: str) -> str:
    """
    Normalizes visually similar punctuation for quote matching.

    This allows exact quote verification to tolerate formatting differences such as:
    - curly apostrophes vs straight apostrophes
    - en dashes/em dashes/nonbreaking hyphens vs normal hyphens
    - nonbreaking spaces vs normal spaces
    - repeated whitespace or line breaks

    This does not allow paraphrases.
    """
    normalized = unicodedata.normalize("NFKC", text)

    for old, new in PUNCTUATION_REPLACEMENTS.items():
        normalized = normalized.replace(old, new)

    return normalize_whitespace(normalized)


def quote_exists_in_text(quote: str, text: str) -> bool:
    """
    Quote verification after safe text normalization.

    Accepts exact substring matches after normalizing whitespace and visually similar
    punctuation. This allows minor formatting differences but does not allow paraphrases.
    """
    normalized_quote = normalize_quote_for_matching(quote)
    normalized_text = normalize_quote_for_matching(text)

    if not normalized_quote:
        return False

    return normalized_quote in normalized_text

def normalize_quote_for_punctuation_insensitive_matching(text: str) -> str:
    """
    Normalizes text for fallback matching when the LLM changes only punctuation.

    This is intended for final report evidence selection, where evidence has already
    been verified earlier. It should not be the primary source-text verification rule.
    """
    normalized = normalize_quote_for_matching(text)
    normalized = normalized.lower()

    # Remove punctuation but keep letters and numbers.
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = normalize_whitespace(normalized)

    return normalized