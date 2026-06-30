import re


def normalize_article_text(text: str) -> str:
    """
    Normalize extracted article text while preserving paragraph boundaries.

    Expected output style:
    paragraph one

    paragraph two
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line.
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty leading/trailing noise but preserve paragraph breaks.
    normalized_lines: list[str] = []
    previous_blank = False

    for line in lines:
        if not line:
            if not previous_blank:
                normalized_lines.append("")
            previous_blank = True
        else:
            normalized_lines.append(line)
            previous_blank = False

    text = "\n".join(normalized_lines).strip()

    # Convert 3+ newlines to exactly 2 newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalize weird internal whitespace, but not paragraph breaks.
    paragraphs = split_paragraphs(text)
    return "\n\n".join(paragraphs)


def split_paragraphs(text: str) -> list[str]:
    """
    Split normalized article text into paragraphs.
    """
    if not text:
        return []

    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    paragraphs = []

    for para in raw_paragraphs:
        para = re.sub(r"[ \t]+", " ", para.strip())
        if para:
            paragraphs.append(para)

    return paragraphs


def count_words(text: str) -> int:
    """
    Simple word count for chunk sizing/debugging.
    """
    if not text:
        return 0
    return len(re.findall(r"\b\S+\b", text))