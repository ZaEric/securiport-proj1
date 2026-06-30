from src.processing.models import SourceDocument, TextChunk

# unimplemented relevance filtering functions, currently just return the input lists unchanged
def filter_sources_by_relevance(
    sources: list[SourceDocument],
    target_name: str,
) -> list[SourceDocument]:
    return sources

# unimplemented relevance filtering functions, currently just return the input lists unchanged
def filter_chunks_by_relevance(
    chunks: list[TextChunk],
    target_name: str,
) -> list[TextChunk]:
    return chunks