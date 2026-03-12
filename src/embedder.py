from typing import List
import numpy as np
 
# Model name — change this to try different embedding models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
 
 
def _load_model():
    """
    Load the sentence transformer model.
    Uses lazy loading so the model is only downloaded once
    and cached for all subsequent calls.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)
 
 
def get_embeddings(chunks: List[str]) -> np.ndarray:
    """
    Generate vector embeddings for a list of text chunks.
 
    Process:
        1. Load the MiniLM model (cached after first download)
        2. Encode all chunks in one batch for efficiency
        3. L2-normalize vectors so cosine similarity = dot product
        4. Return as NumPy array
 
    Args:
        chunks: List of text chunk strings to embed
 
    Returns:
        NumPy array of shape (num_chunks, 384)
        Each row is the embedding vector for one chunk.
 
    Example:
        embeddings = get_embeddings(chunks)
        print(embeddings.shape)   # (72, 384)
    """
    model = _load_model()
 
    embeddings = model.encode(
        chunks,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalize → cosine sim via dot product
        batch_size=32,               # Process 32 chunks at a time
    )
 
    return embeddings
 
 
def get_query_embedding(query: str) -> np.ndarray:
    """
    Generate a vector embedding for a single user query.
 
    Args:
        query: The user's natural language question
 
    Returns:
        NumPy array of shape (1, 384) — one embedding vector
 
    Example:
        vec = get_query_embedding("What is the main topic?")
        print(vec.shape)  # (1, 384)
    """
    model = _load_model()
 
    embedding = model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
 
    return embedding
