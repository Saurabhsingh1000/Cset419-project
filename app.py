# -*- coding: utf-8 -*-
"""
RAG-Based PDF Chatbot — Streamlit Application

A premium, modern Streamlit interface for the RAG-Based PDF Chatbot.
Features:
  - Multi-PDF upload with sidebar controls
  - Configurable Top-K and Temperature sliders
  - Chat interface with conversation history
  - Source attribution (retrieved chunks displayed per answer)
  - Google Gemini API key input
"""

import streamlit as st
from rag_pipeline import ingest_documents, query_pipeline


# -------------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------------
# Custom CSS for a polished, premium look
# -------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---------- Root variables ---------- */
    :root {
        --primary:    #6C63FF;
        --primary-dark: #5148D4;
        --accent:     #00D2FF;
        --bg-dark:    #0E1117;
        --surface:    #1A1D26;
        --surface-2:  #242732;
        --text:       #E8E8ED;
        --text-muted: #9CA3AF;
        --success:    #10B981;
        --border:     #2D3140;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #13151D 0%, #1A1D28 100%);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text) !important;
    }

    /* ---------- Sidebar brand ---------- */
    .sidebar-brand {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .sidebar-brand .logo {
        font-size: 2.8rem;
        margin-bottom: 0.25rem;
    }
    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.25rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #00D2FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sidebar-brand p {
        margin: 0.2rem 0 0 0;
        color: var(--text-muted);
        font-size: 0.8rem;
    }

    /* ---------- Status pills ---------- */
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem 0;
    }
    .status-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-waiting {
        background: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ---------- Main header ---------- */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(135deg, #6C63FF 0%, #00D2FF 50%, #6C63FF 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { background-position: 0% center; }
        50%      { background-position: 200% center; }
    }
    .main-header p {
        color: var(--text-muted);
        font-size: 1rem;
        margin: 0.4rem 0 0 0;
    }

    /* ---------- Chat messages ---------- */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        margin-bottom: 0.8rem !important;
    }

    /* ---------- Source chunks expander ---------- */
    .source-chunk {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        line-height: 1.55;
        color: var(--text);
    }
    .source-chunk .chunk-header {
        font-weight: 600;
        color: var(--accent);
        margin-bottom: 0.35rem;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ---------- Welcome card ---------- */
    .welcome-card {
        background: linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(0,210,255,0.08) 100%);
        border: 1px solid rgba(108,99,255,0.2);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 2rem auto;
        max-width: 650px;
    }
    .welcome-card h3 {
        margin: 0 0 0.5rem 0;
        color: var(--text);
        font-weight: 700;
    }
    .welcome-card p {
        color: var(--text-muted);
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.6;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .feature-item {
        background: rgba(26, 29, 38, 0.6);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
        text-align: left;
    }
    .feature-item .icon {
        font-size: 1.4rem;
        margin-bottom: 0.3rem;
    }
    .feature-item h4 {
        margin: 0;
        font-size: 0.88rem;
        font-weight: 600;
        color: var(--text);
    }
    .feature-item p {
        margin: 0.2rem 0 0 0;
        font-size: 0.78rem;
        color: var(--text-muted);
    }

    /* ---------- Misc ---------- */
    .divider {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------------
# Session state defaults
# -------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False


# -------------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="logo">📄</div>
            <h2>RAG PDF Chatbot</h2>
            <p>Retrieval-Augmented Generation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # API key
    st.markdown("#### 🔑 API Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="Enter your API key…",
        help="Get a free key at https://aistudio.google.com/app/apikey",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # PDF upload
    st.markdown("#### 📁 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to query.",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Retrieval parameters
    st.markdown("#### ⚙️ Retrieval Settings")
    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of document chunks to retrieve per query.",
    )
    temperature = st.slider(
        "LLM Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower = more factual, Higher = more creative.",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Process button
    process_btn = st.button(
        "🚀  Process Documents",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_btn and uploaded_files:
        with st.spinner("Parsing PDFs, chunking, and building vector index…"):
            chunks, index = ingest_documents(uploaded_files)
            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.docs_processed = True
            st.session_state.messages = []  # reset chat when new docs loaded
        st.success(f"✅ Indexed **{len(chunks)}** chunks from {len(uploaded_files)} file(s).")

    # Status indicator
    if st.session_state.docs_processed:
        st.markdown(
            '<div class="status-pill status-ready">● Documents Ready</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill status-waiting">● No Documents Loaded</div>',
            unsafe_allow_html=True,
        )


# -------------------------------------------------------------------------
# Main area
# -------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>RAG PDF Chatbot</h1>
        <p>Upload PDFs and ask questions — answers grounded in your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Welcome card (shown when no messages yet) ----
if not st.session_state.messages and not st.session_state.docs_processed:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome!</h3>
            <p>Upload your PDF documents in the sidebar, enter your Gemini API key,
               and click <b>Process Documents</b> to get started.</p>
            <div class="feature-grid">
                <div class="feature-item">
                    <div class="icon">🔍</div>
                    <h4>Semantic Search</h4>
                    <p>Finds answers by meaning, not keywords</p>
                </div>
                <div class="feature-item">
                    <div class="icon">📎</div>
                    <h4>Source Attribution</h4>
                    <p>See exactly which chunks were used</p>
                </div>
                <div class="feature-item">
                    <div class="icon">🛡️</div>
                    <h4>Grounded Answers</h4>
                    <p>No hallucination — answers from your docs</p>
                </div>
                <div class="feature-item">
                    <div class="icon">⚡</div>
                    <h4>Multi-Document</h4>
                    <p>Process multiple PDFs simultaneously</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---- Render chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source chunks for assistant messages
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 View Source Chunks", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-chunk">
                            <div class="chunk-header">Chunk {src['index'] + 1} · Similarity {src['score']:.3f}</div>
                            {src['chunk']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---- Chat input ----
if user_input := st.chat_input("Ask a question about your documents…"):
    # Validation
    if not st.session_state.docs_processed:
        st.error("⚠️ Please upload and process documents first.")
        st.stop()
    if not api_key:
        st.error("⚠️ Please enter your Google Gemini API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            try:
                answer, sources = query_pipeline(
                    query=user_input,
                    chunks=st.session_state.chunks,
                    index=st.session_state.faiss_index,
                    api_key=api_key,
                    top_k=top_k,
                    temperature=temperature,
                )
                st.markdown(answer)

                # Source attribution
                with st.expander("📎 View Source Chunks", expanded=False):
                    for src in sources:
                        st.markdown(
                            f"""
                            <div class="source-chunk">
                                <div class="chunk-header">Chunk {src['index'] + 1} · Similarity {src['score']:.3f}</div>
                                {src['chunk']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

