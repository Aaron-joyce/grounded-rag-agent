# Grounded Hybrid RAG Agent

An agentic Retrieval-Augmented Generation (RAG) engine combining **Dense Semantic Vector Search** (ChromaDB + `sentence-transformers`) and **Sparse Keyword Search** (BM25) fused via **Reciprocal Rank Fusion (RRF)**. Features strict zero-hallucination guardrails, multi-format PDF/MD ingestion, and an automated evaluation suite.

---

## 🏗️ Architecture Overview

```
                      ┌────────────────────────┐
                      │     User Question      │
                      └───────────┬────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Gemini Agent / Tool Router  │
                   └──────────────┬───────────────┘
                                  │
                                  ▼
                     search_knowledge_base(query)
                                  │
            ┌─────────────────────┴─────────────────────┐
            │                                           │
            ▼                                           ▼
 ┌──────────────────────┐                    ┌──────────────────────┐
 │ Dense Vector Search  │                    │ Sparse BM25 Search   │
 │ (ChromaDB HNSW)      │                    │ (rank_bm25 Okapi)    │
 └──────────┬───────────┘                    └──────────┬───────────┘
            │ Top-10 Chunks                             │ Top-10 Chunks
            └─────────────────────┬─────────────────────┘
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │  Reciprocal Rank Fusion (RRF)   │
                 │  Score = 1/(60+r_v) + 1/(60+r_b)│
                 └────────────────┬────────────────┘
                                  │
                                  ▼
                         Top-5 Fused Chunks
                                  │
                                  ▼
                 ┌─────────────────────────────────┐
                 │ Grounded Answer + Citations     │
                 │       OR Exact Refusal          │
                 └─────────────────────────────────┘
```

---

## 🚀 Key Features

* **Hybrid Search (BM25 + Vector RRF):** Fuses dense embeddings (`all-MiniLM-L6-v2`) with sparse keyword matching (`rank-bm25`) using Reciprocal Rank Fusion ($k=60$), eliminating keyword omission errors.
* **Strict Zero-Hallucination Guardrails:** System prompts enforce strict context locking. If a query is out-of-scope, the agent outputs an exact refusal message instead of hallucinating.
* **Multi-Format Ingestion Pipeline:** Parses `.pdf` (page-by-page via `pypdf`), `.md`, `.txt`, `.py`, `.js`, and `.ts` with sentence-aware chunking.
* **Automated Evaluation Suite (`eval/eval.py`):** Measures Retrieval Recall @ 5, Refusal Compliance Accuracy %, and Query Latency against a benchmark test set.
* **Modular Package Architecture:** Clean directory layout (`src/config.py`, `src/chunker.py`, `src/retriever.py`, `src/indexer.py`, `src/agent.py`) with `main.py` entry point.

---

## 📊 Benchmark Evaluation Results

The evaluation suite (`eval/eval.py`) validates retrieval recall and refusal accuracy against ground-truth and out-of-scope trick queries:

| Metric | Score | Target | Status |
| :--- | :--- | :--- | :--- |
| **Retrieval Recall @ 5** | **100.0%** (4/4) | $\ge$ 85.0% | 🟢 **PASS** |
| **Refusal Accuracy** | **100.0%** (2/2) | 100.0% | 🟢 **PASS** |

### Detailed Evaluation Output

| ID | Type | Question | Expected Source / Action | Retrieved Sources | Status |
|---|---|---|---|---|---|
| 1 | Standard | How do I install this project? | `api.md` | `api.md, providence.pdf` | **PASS** |
| 2 | Standard | What Python version and prerequisites are required? | `api.md` | `api.md, providence.pdf` | **PASS** |
| 3 | Standard | What is Providence in 5 lines? | `providence.pdf` | `providence.pdf` | **PASS** |
| 4 | Standard | What is the market Providence is trying to capture in 5 lines? | `providence.pdf` | `providence.pdf` | **PASS** |
| 5 | Trick | What was Apple's total revenue in 2024? | `REFUSAL` | `providence.pdf` | **PASS** |
| 6 | Trick | How do I configure multi-region Kubernetes clusters on AWS? | `REFUSAL` | `providence.pdf` | **PASS** |

---

## 📦 Project Structure

```
grounded-rag-agent/
├── src/
│   ├── __init__.py          # Package marker
│   ├── config.py            # Model configurations & environment setup
│   ├── chunker.py           # Sentence-aware text chunking logic
│   ├── retriever.py         # ChromaDB HNSW vector store, BM25, & RRF algorithm
│   ├── indexer.py           # Multi-format document parser & indexing pipeline
│   └── agent.py             # Gemini LLM client, tool definition, & grounding prompts
├── docs/                    # Document corpus (.pdf, .md, .txt)
├── eval/
│   ├── test_queries.json    # Benchmark dataset (grounded + trick queries)
│   ├── eval.py              # Automated evaluation test harness
│   └── results.md           # Benchmark summary report
├── main.py                  # Main entry point to index and query
├── pyproject.toml           # Project dependencies & metadata
└── .env.example             # Template for API keys
```

---

## 🛠️ Quickstart & Setup

### Prerequisites
- Python 3.14+
- [`uv`](https://docs.astral.sh/uv/) package manager installed
- A Google Gemini API Key (`GEMINI_API_KEY`)

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/your-username/grounded-rag-agent.git
cd grounded-rag-agent
uv sync
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` and set your key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 3. Index Documents & Run Agent
```bash
uv run main.py
```

### 4. Run Automated Evaluation Suite
```bash
uv run python eval/eval.py
```
