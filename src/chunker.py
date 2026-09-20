"""Sentence-aware text chunking module."""

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into chunks respecting sentence boundaries.

    Args:
        text: Input document text.
        chunk_size: Target size in characters for each chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if end < len(text):
            last_break = max(chunk.rfind(". "), chunk.rfind("? "), chunk.rfind("\n"))
            if last_break > chunk_size // 2:
                end = start + last_break + 1
                chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]
