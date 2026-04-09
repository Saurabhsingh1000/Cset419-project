# -*- coding: utf-8 -*-
"""
RAG Pipeline Module — Core engine for the RAG-Based PDF Chatbot.

Implements:
  - Ingestion Pipeline: PDF parsing, semantic chunking, dense embedding, FAISS indexing
  - Query Pipeline: query embedding, Top-K retrieval, grounded LLM answer generation

Technical specifications (from project documentation):
  - Embedding model: all-MiniLM-L6-v2 (384-dim, L2-normalized)
  - Vector index: FAISS IndexFlatIP (exact inner-product search)
  - Chunk size: 500 chars, overlap: 50 chars, minimum: 30 chars
  - LLM: Google Gemini 1.5 Flash via google-genai SDK
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Fix Windows console encoding (prevents ASCII codec errors with Unicode)
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from google import genai


# ---------------------------------------------------------------------------
# Embedding model (loaded once, reused across calls)
# ---------------------------------------------------------------------------
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Lazy-load the all-MiniLM-L6-v2 sentence transformer model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


# ---------------------------------------------------------------------------
# 1. PDF Parsing
# ---------------------------------------------------------------------------
def parse_pdfs(pdf_files: list) -> str:
    """
    Extract text from one or more PDF files using PyPDF2.

    Each page is prefixed with a [Page X] marker for traceability.
    Returns the full concatenated text corpus.
    """
    corpus_parts: list[str] = []

    for pdf_file in pdf_files:
        reader = PdfReader(pdf_file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                corpus_parts.append(f"[Page {page_num}]\n{text}")

    return "\n\n".join(corpus_parts)


# ---------------------------------------------------------------------------
# 2. Semantic Chunking (Recursive Character Text Splitting)
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_length: int = 30,
) -> list[str]:
    """
    Split *text* into overlapping chunks using a recursive character
    text-splitting strategy.

    Algorithm:
      - Sliding window of *chunk_size* characters with *chunk_overlap* overlap.
      - Attempts to split at sentence boundaries ('. '), then newlines ('\\n'),
        then spaces (' '), falling back to hard character cut.
      - Chunks shorter than *min_chunk_length* are discarded (formatting noise).
    """
    separators = [". ", "\n", " "]
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If we haven't reached the end of the text, try to find a good
        # split point near the end of the window.
        if end < len(text):
            split_pos = -1
            for sep in separators:
                # Search backwards from *end* for the separator within the
                # current window so we don't split mid-sentence.
                pos = text.rfind(sep, start, end)
                if pos != -1 and pos > start:
                    split_pos = pos + len(sep)
                    break

            if split_pos != -1:
                end = split_pos

        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_length:
            chunks.append(chunk)

        # Advance the window by (chunk_size - overlap).
        start += chunk_size - chunk_overlap

    return chunks


# ---------------------------------------------------------------------------
# 3. Dense Vector Embedding
# ---------------------------------------------------------------------------
def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of text strings into L2-normalized 384-dim vectors
    using all-MiniLM-L6-v2.

    Returns an (N, 384) float32 numpy array.
    """
    model = _get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# 4. FAISS Index Construction
# ---------------------------------------------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP index from the embedding matrix.

    IndexFlatIP computes exact Inner Product similarity (equivalent to
    cosine similarity when vectors are L2-normalized).
    """
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# 5. Top-K Retrieval
# ---------------------------------------------------------------------------
def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    top_k: int = 3,
) -> list[dict]:
    """
    Embed the user query, search the FAISS index, and return the Top-K
    most similar chunks with their similarity scores.

    Returns a list of dicts: [{"chunk": str, "score": float, "index": int}]
    """
    model = _get_embedding_model()
    query_vec = model.encode([query], normalize_embeddings=True)
    query_vec = np.array(query_vec, dtype=np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < len(chunks):
            results.append({
                "chunk": chunks[idx],
                "score": float(score),
                "index": int(idx),
            })

    return results


# ---------------------------------------------------------------------------
# 6. Grounded Answer Generation (Google Gemini 1.5 Flash)
# ---------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """You are a precise document question-answering assistant.

**RULES — follow these strictly:**
1. Answer the user's question ONLY using the provided document context below.
2. If the answer is NOT found in the context, respond with: "I don't have enough information in the uploaded document to answer this question."
3. Do NOT use any external knowledge or make assumptions beyond the context.
4. Cite the relevant parts of the context when answering.
5. Be concise yet thorough.

---
DOCUMENT CONTEXT:
{context}
---

USER QUESTION:
{question}
"""


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    api_key: str,
    temperature: float = 0.2,
) -> str:
    """
    Assemble a grounded RAG prompt from the retrieved chunks and send it
    to Google Gemini 1.5 Flash for answer generation.
    """
    # Build the context block from retrieved chunks
    context_parts = []
    for i, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(f"[Chunk {i}] (Similarity: {item['score']:.3f})\n{item['chunk']}")
    context = "\n\n".join(context_parts)

    # Assemble the full prompt
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

    # Call Gemini 1.5 Flash via google.genai SDK
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt,
        config=genai.types.GenerateContentConfig(temperature=temperature),
    )

    return response.text


# ---------------------------------------------------------------------------
# Convenience: Full pipeline wrappers
# ---------------------------------------------------------------------------
def ingest_documents(pdf_files: list) -> tuple[list[str], faiss.IndexFlatIP]:
    """
    Run the full ingestion pipeline:
      PDF files → parse → chunk → embed → FAISS index

    Returns (chunks, faiss_index).
    """
    raw_text = parse_pdfs(pdf_files)
    chunks = chunk_text(raw_text)
    embeddings = embed_texts(chunks)
    index = build_faiss_index(embeddings)
    return chunks, index


def query_pipeline(
    query: str,
    chunks: list[str],
    index: faiss.IndexFlatIP,
    api_key: str,
    top_k: int = 3,
    temperature: float = 0.2,
) -> tuple[str, list[dict]]:
    """
    Run the full query pipeline:
      User query → embed → retrieve Top-K → generate grounded answer

    Returns (answer_text, retrieved_chunks).
    """
    retrieved = retrieve_chunks(query, index, chunks, top_k=top_k)
    answer = generate_answer(query, retrieved, api_key, temperature=temperature)
    return answer, retrieved
