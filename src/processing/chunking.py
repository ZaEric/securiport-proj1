from dataclasses import dataclass
from typing import Any

from src.processing.models import SourceDocument, TextChunk
from src.processing.text_normalization import (
    normalize_article_text,
    split_paragraphs,
    count_words,
)


# -------------------------
# LLM chunking
# -------------------------

@dataclass
class LLMChunkingConfig:
    target_chunk_words: int = 3500
    max_chunk_words: int = 4000
    overlap_paragraphs: int = 1
    chunking_strategy: str = "paragraph_aware"


def chunk_source_document_llm(
    source: SourceDocument,
    config: LLMChunkingConfig | None = None,
) -> list[TextChunk]:
    """
    Paragraph-aware chunking for LLM evidence extraction.

    Current behavior:
    - Normalize article text.
    - Split text into paragraphs.
    - Build chunks up to max_chunk_words.
    - Preserve paragraph boundaries.
    - Optionally overlap paragraph(s) between chunks.
    """
    config = config or LLMChunkingConfig()

    normalized_text = normalize_article_text(source.article_text)
    paragraphs = split_paragraphs(normalized_text)

    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    current_paragraphs: list[str] = []

    i = 0
    while i < len(paragraphs):
        paragraph = paragraphs[i]

        candidate_paragraphs = current_paragraphs + [paragraph]
        candidate_text = "\n\n".join(candidate_paragraphs)
        candidate_word_count = count_words(candidate_text)

        exceeds_limit = (
            current_paragraphs
            and candidate_word_count > config.max_chunk_words
        )

        if exceeds_limit:
            chunks.append(
                _make_chunk(
                    source=source,
                    chunk_index=len(chunks),
                    paragraphs=current_paragraphs,
                )
            )

            current_paragraphs = _get_safe_overlap(
                current_paragraphs=current_paragraphs,
                next_paragraph=paragraph,
                config=config,
            )

            # Retry the same paragraph with safe overlap only.
            continue

        current_paragraphs.append(paragraph)
        i += 1

    if current_paragraphs:
        chunks.append(
            _make_chunk(
                source=source,
                chunk_index=len(chunks),
                paragraphs=current_paragraphs,
            )
        )

    return chunks


def chunk_sources_llm(
    sources: list[SourceDocument],
    config: LLMChunkingConfig | None = None,
) -> list[TextChunk]:
    all_chunks: list[TextChunk] = []

    for source in sources:
        all_chunks.extend(chunk_source_document_llm(source, config=config))

    return all_chunks


def chunking_metadata_llm(config: LLMChunkingConfig) -> dict[str, Any]:
    return {
        "chunking_strategy": config.chunking_strategy,
        "target_chunk_words": config.target_chunk_words,
        "max_chunk_words": config.max_chunk_words,
        "overlap_paragraphs": config.overlap_paragraphs,
    }


# -------------------------
# Classifier chunking
# -------------------------

@dataclass
class ClassifierChunkingConfig:
    window_before: int = 1
    window_after: int = 1
    evidence_unit: str = "target_sentence"
    chunking_strategy: str = "sentence_window"


def chunk_source_document_classifier(
    source: SourceDocument,
    config: ClassifierChunkingConfig | None = None,
) -> list[TextChunk]:
    """
    Future classifier-oriented chunking.

    Intended behavior:
    - Split article text into sentences.
    - Create sentence-window chunks around each target/relevant sentence.
    - The classifier receives neighboring context.
    - The evidence unit remains the target sentence.

    Not implemented yet.
    """
    raise NotImplementedError(
        "Classifier chunking is not implemented yet. "
        "Expected future behavior: sentence-window chunking with target sentence as evidence unit."
    )


def chunk_sources_classifier(
    sources: list[SourceDocument],
    config: ClassifierChunkingConfig | None = None,
) -> list[TextChunk]:
    """
    Future classifier chunking entry point.

    Not implemented yet.
    """
    raise NotImplementedError(
        "Classifier chunking is not implemented yet. "
        "Expected future behavior: sources.json -> sentence-window chunks.json."
    )


def chunking_metadata_classifier(config: ClassifierChunkingConfig) -> dict[str, Any]:
    return {
        "chunking_strategy": config.chunking_strategy,
        "window_before": config.window_before,
        "window_after": config.window_after,
        "evidence_unit": config.evidence_unit,
    }


# -------------------------
# Shared helpers
# -------------------------

def _get_safe_overlap(
    current_paragraphs: list[str],
    next_paragraph: str,
    config: LLMChunkingConfig,
) -> list[str]:
    """
    Return overlap paragraphs only if overlap + next paragraph will fit.

    This prevents an infinite loop where the same overlap paragraph keeps being
    emitted as its own chunk because overlap + next paragraph exceeds max size.
    """
    if config.overlap_paragraphs <= 0:
        return []

    overlap = current_paragraphs[-config.overlap_paragraphs:]

    while overlap:
        candidate_text = "\n\n".join(overlap + [next_paragraph])
        if count_words(candidate_text) <= config.max_chunk_words:
            return overlap

        # Drop the oldest overlap paragraph and try again.
        overlap = overlap[1:]

    return []


def _make_chunk(
    source: SourceDocument,
    chunk_index: int,
    paragraphs: list[str],
) -> TextChunk:
    text = "\n\n".join(paragraphs).strip()

    return TextChunk(
        chunk_id=f"{source.source_id}_chunk_{chunk_index + 1:03d}",
        source_id=source.source_id,
        text=text,
        word_count=count_words(text),
    )