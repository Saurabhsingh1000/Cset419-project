from typing import List
 
 
def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[str]:
    """
    Split a long document string into overlapping text chunks.
 
    Algorithm:
        - Start at position 0
        - Try to cut at a sentence/word boundary near `chunk_size`
        - Move forward by (chunk_size - overlap) to create the next chunk
        - Repeat until the entire text is processed
 
    Args:
        text:       Full document text extracted from PDF
        chunk_size: Maximum number of characters per chunk (default: 500)
        overlap:    Characters shared between consecutive chunks (default: 50)
 
    Returns:
        List of text chunk strings (chunks shorter than 30 chars are removed)
 
    Example:
        chunks = split_into_chunks(text, chunk_size=500, overlap=50)
        print(f"Created {len(chunks)} chunks")
    """
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")
 
    chunks = []
    start  = 0
 
    while start < len(text):
        end = start + chunk_size
 
        # Try to cut at a natural boundary to avoid splitting mid-sentence
        if end < len(text):
            # Prefer sentence boundary (period)
            boundary = text.rfind(".", start, end)
            if boundary == -1 or boundary <= start:
                # Fall back to newline
                boundary = text.rfind("\n", start, end)
            if boundary == -1 or boundary <= start:
                # Fall back to word boundary (space)
                boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1  # include the boundary character
 
        # Extract and clean the chunk
        chunk = text[start:end].strip()
 
        if chunk:
            chunks.append(chunk)
 
        # Slide the window forward with overlap
        start = end - overlap
 
    # Remove very short chunks — likely noise or formatting artifacts
    chunks = [c for c in chunks if len(c) >= 30]
 
    return chunks
 
 
def get_chunk_stats(chunks: List[str]) -> dict:
    """
    Returns statistics about the generated chunks.
 
    Args:
        chunks: List of text chunk strings
 
    Returns:
        Dictionary with count, average length, min and max lengths
    """
    if not chunks:
        return {}
 
    lengths = [len(c) for c in chunks]
    return {
        "total_chunks":   len(chunks),
        "avg_length":     round(sum(lengths) / len(lengths)),
        "min_length":     min(lengths),
        "max_length":     max(lengths),
    }
 
