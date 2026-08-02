import re


def normalize_article_text(text: str) -> str:
    """
    Normalize extracted article text while preserving paragraph boundaries.

    Handles both:
    - blank-line paragraph style: paragraph one\\n\\nparagraph two
    - single-newline paragraph style: paragraph one\\nparagraph two
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line.
    lines = [line.strip() for line in text.split("\n")]

    # Remove leading/trailing empty lines.
    while lines and not lines[0]:
        lines.pop(0)

    while lines and not lines[-1]:
        lines.pop()

    if not lines:
        return ""

    text = "\n".join(lines)

    # Convert 3+ newlines to exactly 2 newlines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # If the extractor produced single-newline paragraph breaks and no blank-line
    # paragraph breaks, treat each non-empty line as a paragraph.
    if "\n\n" not in text and "\n" in text:
        paragraphs = [
            re.sub(r"[ \t]+", " ", line.strip())
            for line in text.split("\n")
            if line.strip()
        ]
        return "\n\n".join(paragraphs)

    # Otherwise, preserve existing blank-line paragraph structure.
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
        # Preserve sentence spacing but remove weird line-internal whitespace.
        para = re.sub(r"[ \t]+", " ", para.strip())
        para = re.sub(r"\n+", " ", para)
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