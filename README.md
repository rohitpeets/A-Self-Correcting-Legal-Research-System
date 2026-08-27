# A-Self-Correcting-Legal-Research-System

Self-correcting RAG system for legal contract Q&A — hybrid (dense+BM25) retrieval, LLM-based query routing, cross-encoder reranking, and self-verification, benchmarked via ablation study on CUAD. Built to go beyond basic RAG toward production-grade retrieval.

---

## Objective

Most RAG tutorials stop at "embed chunks, do a similarity search, generate an answer." This project goes further — combining multiple retrieval strategies, routing each query to the strategy suited to it, reranking the combined output for precision, and (eventually) adding a self-verification and guardrails layer so the system can catch its own bad answers rather than confidently hallucinating.

The dataset is CUAD (Contract Understanding Atticus Dataset), a real-world corpus of commercial legal contracts, used here for retrieval-augmented question answering over contract clauses. The goal isn't just "does retrieval work" but "which retrieval components actually earn their complexity" — measured later via an ablation study rather than assumed.

---

## Tech Stack

- **Language:** Python
- **Embeddings (dense retrieval):** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Query routing:** Groq-hosted LLM (`openai/gpt-oss-20b`) classifying each question into `keyword` / `semantic` / `hybrid`, with a deterministic fallback to `hybrid` on parse failure or API error
- **Reranking:** `sentence-transformers` `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector store:** ChromaDB (`PersistentClient`) — on-disk, avoids re-embedding on every run
- **Keyword search:** `rank_bm25` (`BM25Okapi`)
- **Config:** `python-dotenv` for the Groq API key; `settings.py` for shared retrieval constants
- **Dataset:** CUAD (Contract Understanding Atticus Dataset) — `theatticusproject/cuad-qa` on HuggingFace, requires `datasets<4.0.0` and `trust_remote_code=True`
- **Dev environment:** Windows/PowerShell, VS Code, per-project virtual environments

---

## Features

- **Dense retrieval** — semantic search via sentence embeddings; matches meaning, not just literal keyword overlap
- **BM25 keyword retrieval** — exact-term matching that dense embeddings can blur over (e.g. "Force Majeure," defined terms, section numbers)
- **Reciprocal Rank Fusion (RRF)** — combines dense + BM25 rankings into one fused score per chunk, rewarding chunks both retrievers agree on, without needing to normalize incomparable raw scores (cosine similarity vs. BM25 score)
- **LLM-based query routing** — a Groq-hosted classifier reads each question and decides whether it needs keyword-only, semantic-only, or hybrid retrieval, based on whether its wording is likely to appear verbatim in the contract; falls back to hybrid if the call fails or returns something unparseable
- **Cross-encoder reranking** — takes the routed retriever's top-10 shortlist and rescores each `(query, chunk)` pair jointly, then sorts and returns the final top-5 by that score; deliberately run only on a shortlist since cross-encoders are too slow to run over the full corpus per query
- **Persistent vector storage** — ChromaDB `PersistentClient` writes embeddings to disk once; a `collection.count() == 0` guard prevents re-embedding (and silently duplicating vectors) on every script run
- *(Planned)* Self-verification (LLM-as-judge grounding check), guardrails (token limiter, input/output classifiers), ablation eval harness — see Architecture Roadmap below

---

## Modules

| File | Purpose |
|---|---|
| `dataset_handler.py` | Pulls a sample contract from CUAD and caches it locally, avoiding repeated dataset downloads during development |
| `Chunk.py` | Sentence-based contract chunking (`chunk_contract`) with configurable chunk size and overlap |
| `settings.py` | Shared constants for the retrieval pipeline — `RETRIEVE_N` (per-retriever candidate pool for RRF), `RERANK_CANDIDATES` (shortlist size handed to the cross-encoder), `FINAL_K` (final result count) |
| `embedding.py` | Dense retrieval indexing: embeds chunks into ChromaDB (`embed_and_chunk`) |
| `dense_search.py` | Dense retrieval query: embeds a query and returns the top-`n` chunk IDs by cosine similarity from ChromaDB |
| `bm25_embedding.py` | BM25 keyword retrieval: shared tokenization helpers (`clean_token`/`clean_chunks`/`clean_query`), builds the `BM25Okapi` index, and returns the top-`n` chunk IDs |
| `RRF.py` | Fuses dense and BM25 rank dicts via Reciprocal Rank Fusion (`k=60`), returning the top-`n` chunk IDs by combined relevance |
| `query_classifier.py` | Calls a Groq-hosted LLM with a routing prompt to classify a question as `keyword`, `semantic`, or `hybrid`; parses and validates the response, defaulting to `hybrid` on failure |
| `retrieval.py` | Dispatches to `bm25_search`, `dense_search`, or `rrf` based on the route returned by `query_classifier.classify` |
| `cross_encoder.py` | Reranks the retrieved shortlist using a cross-encoder, scoring `(query, chunk_text)` pairs jointly, then sorts and slices to the final top-`FINAL_K` |
| `app.py` | Entry point — loads and chunks the sample contract, embeds it if not already persisted, takes a question, routes it, retrieves candidates, reranks them, and prints the final ranked results |
| `find_chunk.py` | Debug utility — greps all chunks for a literal term and prints the matches, used to sanity-check chunking/retrieval by hand |
| `test.py` | Evaluation harness for the query router — runs 10 hand-labeled questions through `classify()` and prints PASS/FAIL per case plus an overall score |

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
└───────────┬───────────┘
            ▼
   User question
            │
            ▼
   query_classifier.classify()   → route: keyword | semantic | hybrid
            │
            ▼
   retrieval.retrieve(route)
     ├─ keyword  → bm25_search()
     ├─ semantic → dense_search()
     └─ hybrid   → RRF.rrf()  (dense_search + bm25_search fused)
            │
            ▼
     Top-10 candidate chunk IDs (RERANK_CANDIDATES)
            │
            ▼
   Cross-encoder reranking (cross_encoder.py)
   scores (query, chunk_text) pairs jointly, sorts, slices to FINAL_K
            │
            ▼
     Final top-5 ranked chunks
            │
            ▼
   (Planned: LLM answer generation → self-verification → guardrails)
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
| 6 | Query routing (Groq LLM classifier) | ✅ Done |
| 7 | Self-verification pass (LLM-as-judge grounding check) | ⏳ Planned |
| 8 | Guardrails layer (token limiter, input/output classifiers) | ⏳ Planned |
| 9 | Ablation eval harness (retrieval quality on CUAD) | ⏳ Planned |
| 10 | Dynamic document ingestion | 💭 Stretch goal |
| 11 | Deployment | 💭 Stretch goal |

---

## Confusing Things & Decisions Made

- **Why rank dicts are `{chunk_id: rank}` and not `{rank: chunk_id}`.** The first version stored rank as the key. That's readable for printing an ordered list, but every real operation needed ("what rank did this specific chunk get from each retriever?") requires looking a chunk up by ID — which meant an O(n) scan through the whole dict every time. Flipping to `{chunk_id: rank}` makes that an O(1) `.get()`, which is what RRF fusion does repeatedly. Lesson: pick the dict orientation based on how you'll query it, not how you'll print it.
- **Why RRF treats a missing rank as "contributes 0," not "rank 0."** Early version converted `None` (chunk not retrieved by a ranker) to `0` and then computed `1/(k+0)`, which is a *high* score — implying the chunk beat every actual rank-1 result. That's backwards: a chunk only one retriever found should score *lower* than one both retrievers agree on, not higher. Fixed by skipping the term entirely when a rank is missing, rather than substituting a fake rank.
- **Why `RRF.py` builds a union list before scoring.** Dense and BM25 top-N results aren't identical sets — some chunks appear in only one. To fuse scores you need every chunk that showed up *anywhere*, not just the intersection, so both lists get merged (deduplicated) before scoring.
- **Why the ChromaDB collection has a `collection.count() == 0` guard.** `PersistentClient` writes to disk, so unlike an in-memory client, re-running the embedding step without a guard would call `collection.add()` again with the same chunk IDs — either erroring on duplicate IDs or silently duplicating vectors. The guard means re-embedding only happens once, ever, per persisted collection. Tradeoff: if the source contract changes, the guard doesn't know to invalidate — the `chroma_db` folder has to be cleared manually for now.
- **Why cross-encoder reranking runs on a 10-item shortlist, not all 54+ chunks.** A cross-encoder scores `(query, chunk)` pairs jointly in one forward pass per pair — much more accurate than comparing independently-computed embeddings, but too slow to run against the full corpus on every query. Running it only on the router's already-narrowed shortlist gets bi-encoder speed at scale and cross-encoder accuracy where it counts.
- **Why BM25 needed its own tokenization pipeline (`clean_token`/`clean_chunks`/`clean_query`) instead of reusing raw chunk text.** Unstripped punctuation (e.g. `"Company,"` vs `"Company"`) silently caused BM25 to miss keyword matches it should have caught, since token-for-token string comparison treats them as different words. Fixing this changed the top-5 BM25 results.
- **Why the query router defaults to `hybrid` instead of failing loudly.** A malformed or unparseable LLM response (or a dropped API call) shouldn't take retrieval down entirely — `hybrid` is the safest fallback since it's a superset of what `keyword` or `semantic` alone would retrieve. The router's real accuracy is checked separately via `test.py`'s 10 hand-labeled cases rather than trusted blindly.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install sentence-transformers chromadb rank-bm25 datasets groq python-dotenv nltk
```

> Note: CUAD requires `datasets<4.0.0` for compatibility.
> Query routing requires a `GROQ_API_KEY` in a `.env` file at the project root.

---

## Status

Actively in development. This README is updated as each module is completed — see commit history for granular progress.
