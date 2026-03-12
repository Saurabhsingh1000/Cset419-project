# Cset419-project:

# 📄 RAG-Based PDF Chatbot

> A Retrieval-Augmented Generation (RAG) system that enables intelligent, context-aware question answering over PDF documents using Large Language Models (LLMs).

---

## 📑 Table of Contents

- [Project Description](#-project-description)
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Related Work](#-related-work)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Individual Contributions](#-individual-contributions)
- [Future Work](#-future-work)
- [References](#-references)

---

## 📌 Project Description

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to build a PDF-aware chatbot. The system allows users to upload one or more PDF documents and interact with them via natural language queries. Instead of relying solely on an LLM's parametric memory, the chatbot dynamically retrieves the most relevant chunks from the uploaded PDFs and conditions the LLM's response on that context — greatly reducing hallucination and improving factual accuracy.

The system leverages:
- **PDF Parsing** to extract and clean raw text from documents
- **Semantic Chunking** to split text into meaningful, retrievable units
- **Vector Embeddings** to encode chunks into a high-dimensional semantic space
- **FAISS / ChromaDB** as a vector store for efficient similarity search
- **LLM (e.g., GPT-3.5 / Gemini / LLaMA)** for final answer generation
- **LangChain** as the orchestration framework connecting all components
- **Streamlit** for a clean, user-friendly web interface

---

## ❗ Problem Statement

Large Language Models (LLMs) have demonstrated impressive performance on a wide range of NLP tasks. However, they suffer from two critical limitations in document-centric use cases:

1. **Knowledge cutoff**: LLMs are trained on static datasets and have no awareness of domain-specific or newly created documents.
2. **Hallucination**: LLMs tend to generate plausible-sounding but factually incorrect answers when queried outside their training distribution.

Traditional keyword-based search (e.g., Ctrl+F, grep) cannot handle semantic queries like *"What are the termination conditions mentioned in this contract?"* or *"Summarize the methodology section."*

**The need:** An intelligent system that can ingest arbitrary PDF documents and answer natural language questions grounded in the document's actual content — accurately, quickly, and without hallucination.

---

## ✅ Proposed Solution

We propose a **RAG-based architecture** that decouples knowledge retrieval from language generation:

```
User Query
    │
    ▼
┌─────────────────────┐
│  Query Embedding    │  ← Convert query to vector
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Vector Store      │  ← FAISS / ChromaDB similarity search
│  (PDF chunks)       │
└────────┬────────────┘
         │  Top-K relevant chunks
         ▼
┌─────────────────────┐
│  Prompt Builder     │  ← Combine query + retrieved context
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│       LLM           │  ← Generate grounded answer
│  (GPT / Gemini)     │
└────────┬────────────┘
         │
         ▼
      Answer
```

**Key design decisions:**
- **Chunking strategy**: Recursive character splitting (chunk size = 500, overlap = 50) to preserve sentence context across boundaries
- **Embedding model**: `text-embedding-ada-002` (OpenAI) or `all-MiniLM-L6-v2` (HuggingFace) for semantic similarity
- **Retriever**: Top-3 chunks via cosine similarity
- **LLM prompt**: System-level instruction to answer only from provided context, falling back to *"I don't know"* if no relevant chunk is found

---

## 📚 Related Work

The following research papers directly informed the design and methodology of this project:

| # | Paper | Authors | Year | Contribution to This Project |
|---|-------|---------|------|------------------------------|
| 1 | **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | Lewis et al. | 2020 | Foundational RAG architecture; combining retrieval with seq2seq generation |
| 2 | **Dense Passage Retrieval for Open-Domain Question Answering** | Karpukhin et al. | 2020 | DPR-based dense retrieval; bi-encoder design for query-document matching |
| 3 | **REALM: Retrieval-Augmented Language Model Pre-Training** | Guu et al. | 2020 | Joint training of retriever and reader; motivates the use of neural retrievers |
| 4 | **Improving language models by retrieving from trillions of tokens** (RETRO) | Borgeaud et al. | 2022 | Large-scale chunked cross-attention retrieval; chunk-level retrieval design |
| 5 | **LangChain: Building Applications with LLMs through Composability** | Chase | 2022 | Framework used for chaining retriever, prompt, and LLM in our pipeline |
| 6 | **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** | Reimers & Gurevych | 2019 | Basis for semantic similarity; informed our choice of embedding model |
| 7 | **FAISS: A Library for Efficient Similarity Search** | Johnson et al. | 2019 | Efficient nearest-neighbor vector search used in our vector store |
| 8 | **Hallucination in Large Language Models: A Survey** | Ji et al. | 2023 | Motivates the use of RAG to ground LLM responses in source documents |

### Summary of Related Approaches

**Naive LLM QA** (e.g., plain ChatGPT): No document grounding; high hallucination risk.

**BM25 / TF-IDF retrieval**: Keyword-based; fails on semantically paraphrased queries.

**RAG (Lewis et al., 2020)**: Our approach directly extends this — we apply it to the PDF domain with chunking and modern embedding models.

**Fine-tuning on domain data**: Expensive, requires labeled data, not generalized for arbitrary PDFs. RAG is more flexible and cost-effective.

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                   │
│                                                          │
│  PDF File → PyPDF2/pdfplumber → Text Cleaning →          │
│  RecursiveCharacterTextSplitter → Embeddings →           │
│  FAISS / ChromaDB Vector Store                           │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                        │
│                                                          │
│  User Query → Embedding → Similarity Search (Top-K) →   │
│  Prompt Construction → LLM → Response                   │
└──────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 📤 Upload single or multiple PDF files
- 💬 Ask natural language questions about uploaded documents
- 🔍 Semantic search over document chunks using vector embeddings
- 🤖 LLM-generated answers grounded in retrieved context
- 🧠 Conversation memory (multi-turn chat support)
- 📊 Source citation — shows which PDF chunk was used for each answer
- 🌐 Streamlit web UI for easy interaction

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| PDF Parsing | PyPDF2, pdfplumber |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | OpenAI `text-embedding-ada-002` / HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS / ChromaDB |
| LLM | OpenAI GPT-3.5-turbo / Google Gemini / Ollama (LLaMA) |
| Orchestration | LangChain |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 👥 Individual Contributions

| Member | Roll No. | Contributions |
|---|---|---|
| **[Saurabh Kumar Singh]** | [E23CSEU2018] | Project architecture design, LangChain pipeline, vector store integration, README |
| **[Abhishek Khaiwal]** | [E23CSEU2041] | PDF parsing module, text preprocessing, chunking strategy |
| **[Khoushik Vadde]** | [E23CSEU2020] | Streamlit frontend UI, user testing, deployment |
| **[Ayush Kumar Jha]** | [E23CSEU2047] | Experiment design, RAGAS evaluation, results analysis, presentation |


---

## 🔮 Future Work

- [ ] Support for scanned PDFs via OCR (Tesseract integration)
- [ ] Multi-modal RAG: image and table extraction from PDFs
- [ ] Hybrid retrieval: combine BM25 sparse + FAISS dense retrieval (RRF reranking)
- [ ] Fine-tuned embedding model on domain-specific corpora
- [ ] Deploy as a REST API with FastAPI for production use
- [ ] Add support for other document formats (.docx, .pptx, .csv)

---

## 📁 Repository Structure

```
rag-pdf-chatbot/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
├── README.md                 # This file
│
├── src/
│   ├── pdf_loader.py         # PDF parsing and text extraction
│   ├── chunker.py            # Text splitting logic
│   ├── embedder.py           # Embedding generation
│   ├── vector_store.py       # FAISS/ChromaDB interface
│   ├── retriever.py          # Similarity search
│   ├── llm_chain.py          # LangChain QA chain
│   └── prompt_template.py    # LLM prompt construction
│
├── data/
│   ├── sample_pdfs/          # Sample PDFs for testing
│   └── eval_dataset/         # QA pairs for evaluation
│
├── experiments/
│   ├── eval_ragas.py         # RAGAS evaluation script
│   ├── results/              # Experiment result logs
│   └── config.yaml           # Experiment configuration
│
└── docs/
    ├── architecture.png      # System architecture diagram
    └── demo.gif              # Demo GIF of the app
```

---

## 📜 References

1. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. https://arxiv.org/abs/2005.11401
2. Karpukhin, V., et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020. https://arxiv.org/abs/2004.04906
3. Guu, K., et al. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training.* ICML 2020. https://arxiv.org/abs/2002.08909
4. Borgeaud, S., et al. (2022). *Improving language models by retrieving from trillions of tokens.* ICML 2022. https://arxiv.org/abs/2112.04426
5. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019. https://arxiv.org/abs/1908.10084
6. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. https://arxiv.org/abs/1702.08734
7. Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys. https://arxiv.org/abs/2202.03629
8. Es, S., et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* https://arxiv.org/abs/2309.15217

---
