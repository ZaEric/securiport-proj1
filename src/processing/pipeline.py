from typing import Any

from src.processing.models import SourceDocument, TextChunk
from src.processing.chunking import (
    LLMChunkingConfig,
    ClassifierChunkingConfig,
    chunk_sources_llm,
    chunk_sources_classifier,
    chunking_metadata_llm,
    chunking_metadata_classifier,
)
from src.processing.relevance import (
    filter_sources_by_relevance,
    filter_chunks_by_relevance,
)


def process_sources_for_nlp_llm(
    raw_sources: list[dict[str, Any]],
    run_id: str,
    target_name: str,
    config: LLMChunkingConfig | None = None,
) -> dict[str, Any]:
    """
    Processing-stage entry point for LLM evidence extraction.

    Input:
    - sources.json sources list

    Output:
    - chunks.json-compatible dictionary using paragraph-aware chunks
    """
    config = config or LLMChunkingConfig()

    sources = [_source_from_dict(item) for item in raw_sources]

    relevant_sources = filter_sources_by_relevance(
        sources=sources,
        target_name=target_name,
    )

    chunks = chunk_sources_llm(
        sources=relevant_sources,
        config=config,
    )

    relevant_chunks = filter_chunks_by_relevance(
        chunks=chunks,
        target_name=target_name,
    )

    return {
        "run_id": run_id,
        "target_name": target_name,
        "chunks": chunks_to_dicts(relevant_chunks),
        "metadata": chunking_metadata_llm(config),
    }


def process_sources_for_nlp_classifier(
    raw_sources: list[dict[str, Any]],
    run_id: str,
    target_name: str,
    config: ClassifierChunkingConfig | None = None,
) -> dict[str, Any]:
    """
    Processing-stage entry point for classifier evidence extraction.

    Not implemented yet. Reserved so API/design can point to the expected future path.
    """
    config = config or ClassifierChunkingConfig()

    sources = [_source_from_dict(item) for item in raw_sources]

    relevant_sources = filter_sources_by_relevance(
        sources=sources,
        target_name=target_name,
    )

    chunks = chunk_sources_classifier(
        sources=relevant_sources,
        config=config,
    )

    relevant_chunks = filter_chunks_by_relevance(
        chunks=chunks,
        target_name=target_name,
    )

    return {
        "run_id": run_id,
        "target_name": target_name,
        "chunks": chunks_to_dicts(relevant_chunks),
        "metadata": chunking_metadata_classifier(config),
    }


def chunks_to_dicts(chunks: list[TextChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "text": chunk.text,
            "word_count": chunk.word_count,
        }
        for chunk in chunks
    ]


def _source_from_dict(item: dict[str, Any]) -> SourceDocument:
    return SourceDocument(
        source_id=item["source_id"],
        url=item["url"],
        title=item.get("title", ""),
        article_text=item["article_text"],
        metadata={
            key: value
            for key, value in item.items()
            if key not in {"source_id", "url", "title", "article_text"}
        },
    )