# A-Self-Correcting-Legal-Research-System

Self-correcting RAG system for legal contract Q&A — hybrid (dense+BM25) retrieval, cross-encoder reranking, query routing, and self-verification, benchmarked via ablation study on CUAD. Built to go beyond basic RAG toward production-grade retrieval.

---

## Objective

Most RAG tutorials stop at "embed chunks, do a similarity search, generate an answer." This project goes further — combining multiple retrieval strategies, reranking their combined output for precision, and (eventually) adding a self-verification and guardrails layer so the system can catch its own bad answers rather than confidently hallucinating.

The dataset is CUAD (Contract Understanding Atticus Dataset), a real-world corpus of commercial legal contracts, used here for retrieval-augmented question answering over contract clauses. The goal isn't just "does retrieval work" but "which retrieval components actually earn their complexity" — measured later via an ablation study rather than assumed.

---

## Tech Stack

- **Language:** Python
- **Embeddings (dense retrieval):** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Reranking:** `sentence-transformers` `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector store:** ChromaDB (`PersistentClient`) — on-disk, avoids re-embedding on every run
- **Keyword search:** `rank_bm25` (`BM25Okapi`)
- **Dataset:** CUAD (Contract Understanding Atticus Dataset) — `theatticusproject/cuad-qa` on HuggingFace, requires `datasets<4.0.0` and `trust_remote_code=True`
- **Dev environment:** Windows/PowerShell, VS Code, per-project virtual environments

---

## Features

- **Dense retrieval** — semantic search via sentence embeddings; matches meaning, not just literal keyword overlap
- **BM25 keyword retrieval** — exact-term matching that dense embeddings can blur over (e.g. "Force Majeure," defined terms, section numbers)
- **Reciprocal Rank Fusion (RRF)** — combines dense + BM25 rankings into one fused score per chunk, rewarding chunks both retrievers agree on, without needing to normalize incomparable raw scores (cosine similarity vs. BM25 score)
- **Cross-encoder reranking** — takes the RRF top-10 shortlist and rescores each `(query, chunk)` pair jointly for higher precision than either retriever alone; deliberately run only on a shortlist since cross-encoders are too slow to run over the full corpus per query
- **Persistent vector storage** — ChromaDB `PersistentClient` writes embeddings to disk once; a `collection.count() == 0` guard prevents re-embedding (and silently duplicating vectors) on every script run
- *(Planned)* Query routing, self-verification (LLM-as-judge grounding check), guardrails (token limiter, input/output classifiers), ablation eval harness — see Architecture Roadmap below

---

## Modules

| File | Purpose |
|---|---|
| `dataset_handler.py` | Pulls a sample contract from CUAD and caches it locally, avoiding repeated dataset downloads during development |
| `Chunk.py` | Sentence-based contract chunking (`chunk_contract`) with configurable chunk size and overlap |
| `parasrc.py` | Shared setup: loads the sample contract, runs chunking, loads the embedding model, initializes the persistent ChromaDB client/collection, and defines the test `query` used across modules |
| `embedding.py` | Dense retrieval: embeds chunks into ChromaDB (`embed_and_chunk`, guarded against re-embedding), searches by cosine similarity (`dense_search`), and builds `dense_ranks` as a `{chunk_id: rank}` dict |
| `bm25_embedding.py` | BM25 keyword retrieval: shared tokenization helpers (`clean_token`/`clean_chunks`/`clean_query`), builds the `BM25Okapi` index, and produces `bm_ranks` as a `{chunk_id: rank}` dict |
| `RRF.py` | Fuses `dense_ranks` and `bm_ranks` via Reciprocal Rank Fusion (`k=60`), returning the top-10 chunk IDs by combined relevance |
| `cross_encoder.py` | Reranks the RRF top-10 shortlist using a cross-encoder, scoring `(query, chunk_text)` pairs jointly for the final relevance ordering |
| `test.py` | Quick sanity check — prints `collection.count()` to confirm the persisted ChromaDB collection hasn't been duplicated |

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
(dense vectors,        (keyword statistics)
 persisted to disk)
│                       │
▼                       ▼
dense_search(query)    bm25_search(query)
→ {chunk_id: rank}     → {chunk_id: rank}
│                       │
└───────────┬───────────┘
            ▼
   Reciprocal Rank Fusion (RRF.py)
   score = 1/(k+rank_dense) + 1/(k+rank_bm25)
            │
            ▼
     Top-10 fused chunk IDs
            │
            ▼
   Cross-encoder reranking (cross_encoder.py)
   scores (query, chunk_text) pairs jointly
            │
            ▼
     Final top-5 ranked chunks
            │
            ▼
   (Planned: query routing → LLM answer generation →
    self-verification → guardrails)
```

---

## Architecture Roadmap

| # | Stage | Status |
|---|---|---|
| 1 | Chunking | ✅ Done |
| 2 | Dense retrieval (embeddings + ChromaDB) | ✅ Done |
| 3 | BM25 keyword retrieval | ✅ Done |
| 4 | Hybrid retrieval — dense + BM25 via RRF | ✅ Done |
| 5 | Cross-encoder reranking | ✅ Done |
| 6 | Query routing | ⏳ Planned |
| 7 | Self-verification pass (LLM-as-judge grounding check) | ⏳ Planned |
| 8 | Guardrails layer (token limiter, input/output classifiers) | ⏳ Planned |
| 9 | Ablation eval harness | ⏳ Planned |
| 10 | Dynamic document ingestion | 💭 Stretch goal |
| 11 | Deployment | 💭 Stretch goal |

---

## Confusing Things & Decisions Made

- **Why rank dicts are `{chunk_id: rank}` and not `{rank: chunk_id}`.** The first version stored rank as the key. That's readable for printing an ordered list, but every real operation needed ("what rank did this specific chunk get from each retriever?") requires looking a chunk up by ID — which meant an O(n) scan through the whole dict every time. Flipping to `{chunk_id: rank}` makes that an O(1) `.get()`, which is what RRF fusion does repeatedly. Lesson: pick the dict orientation based on how you'll query it, not how you'll print it.
- **Why RRF treats a missing rank as "contributes 0," not "rank 0."** Early version converted `None` (chunk not retrieved by a ranker) to `0` and then computed `1/(k+0)`, which is a *high* score — implying the chunk beat every actual rank-1 result. That's backwards: a chunk only one retriever found should score *lower* than one both retrievers agree on, not higher. Fixed by skipping the term entirely when a rank is missing, rather than substituting a fake rank.
- **Why `RRF.py` builds a union list before scoring.** Dense and BM25 top-5 results aren't identical sets — some chunks appear in only one. To fuse scores you need every chunk that showed up *anywhere*, not just the intersection, so both lists get merged (deduplicated) before scoring.
- **Why the ChromaDB collection has a `collection.count() == 0` guard.** `PersistentClient` writes to disk, so unlike an in-memory client, re-running `embedding.py` without a guard would call `collection.add()` again with the same chunk IDs — either erroring on duplicate IDs or silently duplicating vectors. The guard means re-embedding only happens once, ever, per persisted collection. Tradeoff: if the source contract changes, the guard doesn't know to invalidate — the `chroma_db` folder has to be cleared manually for now.
- **Why cross-encoder reranking runs on a 10-item shortlist, not all 54+ chunks.** A cross-encoder scores `(query, chunk)` pairs jointly in one forward pass per pair — much more accurate than comparing independently-computed embeddings, but too slow to run against the full corpus on every query. Running it only on RRF's already-narrowed shortlist gets bi-encoder speed at scale and cross-encoder accuracy where it counts.
- **Why BM25 needed its own tokenization pipeline (`clean_token`/`clean_chunks`/`clean_query`) instead of reusing raw chunk text.** Unstripped punctuation (e.g. `"Company,"` vs `"Company"`) silently caused BM25 to miss keyword matches it should have caught, since token-for-token string comparison treats them as different words. Fixing this changed the top-5 BM25 results.

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