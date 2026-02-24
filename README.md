# Industry-Grade Custom RAG System

A robust, asynchronous Retrieval-Augmented Generation (RAG) system built entirely from scratch in Python, utilizing standard open-source libraries without relying on high-level abstractions like LangChain or LlamaIndex.

## 🚀 Features

*   **Asynchronous Architecture**: Built extensively on `asyncio` for non-blocking document ingestion, embedding generation, and LLM querying, making it highly efficient.
*   **Modular Design**: Uses Python Abstract Base Classes (`abc`) to ensure components (parsers, chunkers, retrievers, generators) are completely swappable.
*   **Multi-Format Document Ingestion**: Native parsers for `PDF`, `DOCX`, `TXT`, and `MD` files.
*   **Intelligent Chunking**: Sliding window chunker with exact token boundaries (via `tiktoken`) that preserves cross-boundary semantics and automatically attaches metadata (source, page approximations, timestamps).
*   **Advanced Embedding Engine**: Uses `sentence-transformers` for batch embeddings with automatic L2-normalization to ensure FAISS cosine similarity compatibility.
*   **Hybrid Retrieval Pipeline**:
    *   **Dense Retrieval**: FAISS HNSW indexing for fast approximate nearest neighbor search.
    *   **Sparse Retrieval**: BM25 keyword matching via `rank_bm25`.
    *   **Reciprocal Rank Fusion (RRF)**: Mathematically fuses dense and sparse scores to maximize recall.
*   **Cross-Encoder Reranking**: Re-orders the fused retrieved context using `ms-marco-MiniLM-L-6-v2` for absolute semantic relevance.
*   **Dynamic Query Transformations**: Features a `QueryPipeline` that expands user queries using:
    *   **HyDE (Hypothetical Document Embeddings)**
    *   **Multi-Query Decomposition**
*   **Token-Aware Generation**: Dynamically monitors context token budgets before prompting the LLM, proactively discarding overflowing chunks while maintaining source citations. Supports both OpenAI and Google Gemini models with streaming yields.
*   **Interactive Web UI**: A clean, persistent chat interface built with Streamlit for uploading files, configuring retrieval algorithms, and interacting with the RAG pipeline.

## 🛠️ Technology Stack

*   **Vector Database**: `faiss-cpu`
*   **Embeddings & Reranking**: `sentence-transformers`
*   **Sparse Retrieval**: `rank_bm25`
*   **LLM Providers**: `google-genai` (Gemini), `openai`
*   **Token Counting**: `tiktoken`
*   **Document Parsing**: `pypdf`, `python-docx`
*   **Web Framework**: `streamlit`
*   **Logging**: `loguru`
*   **Testing**: `pytest`

## 📦 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd Rag
    ```

2.  **Install dependencies:**
    This project is optimized for Python 3.9 on Windows.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    GEMINI_API_KEY=your_gemini_api_key_here
    OPENAI_API_KEY=your_openai_api_key_here  # Optional, if using OpenAI models
    ```

4.  **Configure System Parameters:**
    Adjust hyperparameters (chunk sizes, retrieval top-k, model selections) in `config.yaml`.

## 🎮 Running the Application

Start the Streamlit web interface:
```bash
python -m streamlit run app.py
```
*Note for Windows/Python 3.9 users: The system is configured to use the polling file watcher (`.streamlit/config.toml`) to prevent watchdog path drive comparison errors during hot-reloads.*

## 🧪 Testing

The codebase includes a comprehensive `pytest` suite testing isolation boundaries across ingestion, embeddings, vector stores, and retrieval mechanisms.

Run all tests:
```bash
pytest
```

## 🏗️ Project Structure

```text
Rag/
├── app.py                      # Main Streamlit web application
├── config.yaml                 # System configuration and hyperparameters
├── requirements.txt            # Pinned Python dependencies
├── .env                        # Environment variables (API Keys)
├── .streamlit/                 # Streamlit specific configurations
│   └── config.toml
├── data/                       # Persistent storage for serialized FAISS/BM25 indexes
├── src/                        # Core application source code
│   ├── config.py               # YAML configuration loader
│   ├── logger.py               # Loguru JSON logger setup
│   ├── pipeline.py             # Main Orchestrator integrating all RAG components
│   ├── ingestion/              # Parsers, Chunkers, and Base classes
│   ├── embeddings/             # Sentence-transformers embedding engine
│   ├── evaluation/             # LLM-as-a-judge faithfulness evaluators
│   ├── generation/             # Context-aware prompt builders and LLM generators
│   ├── retrieval/              # Dense/Sparse stores, RRF, HyDE, and Rerankers
│   └── vector_store/           # FAISS HNSW implementation
└── tests/                      # Pytest verification suites
```

## 🐛 Known Environmental Quirks (Windows)
*   **Sophos Web Appliance & SSL**: If running behind a strict corporate firewall, ensure your network allows traffic to `generativelanguage.googleapis.com` or `api.openai.com`. Initial testing utilized an unverified SSL bypass which triggered security intercept pages (HTML 502s) from the Sophos appliance. The code natively uses standard verified HTTPS requests to prevent this.
*   **Rust DLL Load Failures**: If encountered upon importing cryptographic libraries on Windows, ensure the `cryptography` package remains pinned to version `41.0.7` as defined in `requirements.txt`.
