# A-Self-Correcting-Legal-Research-System

Self-correcting RAG system for legal contract Q&A — hybrid (dense+BM25) retrieval, cross-encoder reranking, query routing, and self-verification, benchmarked via ablation study on CUAD. Built to go beyond basic RAG toward production-grade retrieval.

---

## Overview

Most RAG tutorials stop at "embed chunks, do a similarity search, generate an answer." This project is built to go further — combining multiple retrieval strategies, adding a safety/guardrails layer, and measuring which components actually contribute value (rather than assuming more complexity = better results).

The dataset is CUAD (Contract Understanding Atticus Dataset), a real-world corpus of commercial legal contracts, used here for retrieval-augmented question answering over contract clauses.

---

## Progress

### ✅ Module 1 — Chunking
- Sentence-based contract chunking with configurable overlap (`chunk_contract`)
- Tested on CUAD (`theatticusproject/cuad-qa`) — 54 chunks generated from a 216-sentence sample contract, with 1-sentence overlap between chunks
- Sample contract cached locally (`data/processed/sample_contract.txt`) to avoid repeated dataset downloads during development

### ✅ Module 2 — Dense Retrieval
- Chunks embedded using `sentence-transformers` (`all-MiniLM-L6-v2`), producing `(54, 384)` vectors
- Stored in ChromaDB via `PersistentClient` for on-disk persistence across runs
- `embed_and_chunk(chunks, model, collection)` — embeds and stores chunks
- `dense_search(query, model, collection, n_results=5)` — returns top-N chunk IDs ranked by cosine similarity
- **Validated on an out-of-vocabulary query:** asking "can the tenant terminate the lease early?" against a document that is actually a *distributor agreement* (no "tenant" or "lease" anywhere in the text) still correctly retrieved the relevant termination clauses — confirming the system matches on **meaning**, not literal keyword overlap

### ✅ Module 3 — BM25 Keyword Search
- Keyword-based retrieval using `rank_bm25` (`BM25Okapi`) to complement dense search
- Shared tokenization pipeline (`clean_token`, `clean_chunks`, `clean_query`) — lowercases and strips punctuation so chunks and queries tokenize consistently
- `bm25_search(query, bm25, chunks, n_results=5)` — returns top-N chunk IDs ranked by keyword relevance
- Fixed a tokenization bug where unstripped punctuation (e.g. `"Company,"` vs `"Company"`) silently caused missed keyword matches — correcting it changed the top-5 results, surfacing previously-missed relevant chunks

### 🔄 Module 4 — In Progress: Hybrid Retrieval (Reciprocal Rank Fusion)
- Combining dense and BM25 rankings into a single fused ranking using RRF
- Both retrieval functions now return **chunk IDs** (not raw text) to keep merging lightweight and avoid comparing full paragraphs
- Building `build_rank_dict()` to map `{chunk_id: rank}` per retrieval method before fusing scores with the formula:
  `RRF_score(chunk) = 1/(k + rank_dense) + 1/(k + rank_bm25)`, with `k=60` (standard default from the original RRF paper)

---

## Data Flow (current pipeline)

```
Raw contract text (CUAD)
        │
        ▼
  chunk_contract()          → sentence-based chunks w/ overlap
        │
        ▼
┌───────────────┬────────────────┐
│               │                │
▼               ▼                
embed_and_chunk()      clean_chunks()
(sentence-transformers) (lowercase + strip punctuation)
│                       │
▼                       ▼
ChromaDB               BM25Okapi index
(dense vectors)        (keyword statistics)
│                       │
▼                       ▼
dense_search(query)    bm25_search(query)
→ ranked chunk IDs     → ranked chunk IDs
│                       │
└───────────┬───────────┘
            ▼
   Reciprocal Rank Fusion   ← in progress
            │
            ▼
     Final ranked chunks
            │
            ▼
   (Planned: cross-encoder reranking → query routing →
    LLM answer generation → self-verification → guardrails)
```

---

## Architecture Roadmap

| # | Stage | Status |
|---|---|---|
| 1 | Chunking | ✅ Done |
| 2 | Dense retrieval (embeddings + ChromaDB) | ✅ Done |
| 3 | Hybrid retrieval — dense + BM25 via RRF | 🔄 In progress |
| 4 | Cross-encoder reranking | ⏳ Planned |
| 5 | Query routing | ⏳ Planned |
| 6 | Self-verification pass (LLM-as-judge grounding check) | ⏳ Planned |
| 7 | Guardrails layer (token limiter, input/output classifiers) | ⏳ Planned |
| 8 | Ablation eval harness | ⏳ Planned |
| 9 | Dynamic document ingestion | 💭 Stretch goal |
| 10 | Deployment | 💭 Stretch goal |

---

## Why Hybrid Retrieval?

Dense embeddings and keyword search (BM25) each cover a blind spot the other one has:

- **Dense retrieval** understands *meaning* — it can match a paraphrased question to a relevant chunk even with zero shared vocabulary (see Module 2's tenant/lease example above). Its weakness: it can blur over exact, legally significant terms (e.g. "Force Majeure," specific section numbers, defined terms).
- **BM25** has no understanding of meaning at all — it purely counts weighted word overlap. Its strength is nailing exact terms; its weakness is missing paraphrases entirely.

Combining both via Reciprocal Rank Fusion means a chunk that's relevant either semantically *or* lexically — ideally both — has a strong chance of surfacing in the final ranked results. This mirrors how production-grade RAG systems are typically designed, rather than relying on a single retrieval strategy.

---

## Guardrails (Planned)

Independent of the self-verification pass, a defense-in-depth guardrails layer is planned:
- **Token limiter** — caps input/output length for cost control and to prevent context-window abuse
- **Pre-LLM input classifier** — detects prompt injection and off-topic/malicious queries before retrieval runs
- **Post-LLM output classifier** — checks generated answers for leakage, toxicity, or policy violations (a separate mechanism from self-verification's grounding check, catching different failure modes)

---

## Tech Stack

- **Language:** Python
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (`PersistentClient`)
- **Keyword search:** `rank_bm25` (`BM25Okapi`)
- **Dataset:** CUAD (Contract Understanding Atticus Dataset) — `theatticusproject/cuad-qa` on HuggingFace, requires `datasets<4.0.0` and `trust_remote_code=True`
- **Dev environment:** Windows/PowerShell, VS Code, per-project virtual environments

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install sentence-transformers chromadb rank_bm25 datasets
```

> Note: CUAD requires `datasets<4.0.0` for compatibility.

---

## Status

Actively in development. This README is updated as each module is completed — see commit history for granular progress.
