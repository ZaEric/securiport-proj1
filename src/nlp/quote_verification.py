import re


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def quote_exists_in_text(quote: str, text: str) -> bool:
    """
    MVP quote verification.

    Accepts exact substring match after whitespace normalization.
    This allows line breaks or repeated spaces to differ, but does not allow paraphrases.
    """
    normalized_quote = normalize_whitespace(quote)
    normalized_text = normalize_whitespace(text)

    if not normalized_quote:
        return False

    return normalized_quote in normalized_text