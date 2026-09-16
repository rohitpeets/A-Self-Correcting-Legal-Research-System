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
- **LLM calls (Groq-hosted):** query routing and grounding verification via `openai/gpt-oss-20b`, generation via `openai/gpt-oss-120b`, input-safety screening via `meta-llama/llama-prompt-guard-2-86m` — all through the `groq` SDK
- **Config:** `python-dotenv` for the Groq API key; `settings.py` for shared retrieval/generation constants
- **Dataset:** CUAD (Contract Understanding Atticus Dataset) — `theatticusproject/cuad-qa` on HuggingFace, requires `datasets<4.0.0` and `trust_remote_code=True`
- **Dev environment:** Windows/PowerShell, VS Code, per-project virtual environments

---

## Features

- **Dense retrieval** — semantic search via sentence embeddings; matches meaning, not just literal keyword overlap
- **BM25 keyword retrieval** — exact-term matching that dense embeddings can blur over (e.g. "Force Majeure," defined terms, section numbers)
- **Reciprocal Rank Fusion (RRF)** — combines dense + BM25 rankings into one fused score per chunk, rewarding chunks both retrievers agree on, without needing to normalize incomparable raw scores (cosine similarity vs. BM25 score)
- **Cross-encoder reranking** — takes the RRF top-10 shortlist and rescores each `(query, chunk)` pair jointly for higher precision than either retriever alone; deliberately run only on a shortlist since cross-encoders are too slow to run over the full corpus per query
- **Persistent vector storage** — ChromaDB `PersistentClient` writes embeddings to disk once; a `collection.count() == 0` guard prevents re-embedding (and silently duplicating vectors) on every script run
- **Query routing** — an LLM classifies each question as `keyword`, `semantic`, or `hybrid` before retrieval, so literal-term questions (section numbers, defined terms) use BM25 and everyday-language questions use dense/hybrid search instead of always paying for the most expensive strategy
- **LLM answer generation with citations** — answers are generated only from the reranked shortlist, with every factual claim required to cite its passage number; the model abstains with `NOT_FOUND_IN_CONTRACT` when the passages don't support an answer
- **Self-verification (LLM-as-judge grounding check)** — a second, independent LLM call checks whether the generated answer is actually supported by its cited passages; a failed check overrides the answer to abstain even if generation didn't catch its own error
- **Input guardrails** — a length cap plus Groq's dedicated Prompt Guard model screen every query for prompt-injection/jailbreak attempts before retrieval or generation ever run; a tripped guardrail abstains the same way an unanswerable question does
- **Ablation eval harness** — runs CUAD-QA's own gold question/answer pairs for the sample contract through all three retrieval strategies and reports retrieval recall, abstain accuracy, and citation accuracy per strategy, so retrieval choices are measured rather than assumed

---

## Modules

| File | Purpose |
|---|---|
| `dataset_handler.py` | Pulls a sample contract from CUAD and caches it locally (`data/processed/sample_contract.txt`), avoiding repeated dataset downloads during development |
| `Chunk.py` | Sentence-based contract chunking (`chunk_contract`) with configurable chunk size and overlap |
| `settings.py` | Central config: retrieval/rerank sizes (`RETRIEVE_N`, `RERANK_CANDIDATES`, `FINAL_K`), model names for routing/generation/verification, and the `NOT_FOUND` sentinel the rest of the pipeline checks against |
| `embedding.py` | Embeds chunks into ChromaDB (`embed_and_chunk`); pairs with the `collection.count() == 0` guard in `app.py` so re-running doesn't duplicate vectors |
| `dense_search.py` | Dense retrieval: embeds a query and returns ChromaDB's nearest chunk IDs by cosine similarity (`dense_search`) |
| `bm25_embedding.py` | BM25 keyword retrieval: shared tokenization helpers (`clean_token`/`clean_chunks`/`clean_query`), builds a `BM25Okapi` index per call, and returns ranked chunk IDs (`bm25_search`) |
| `RRF.py` | Fuses dense and BM25 rankings via Reciprocal Rank Fusion (`k=60`) into one fused top-N list of chunk IDs (`rrf`) |
| `retrieval.py` | Single retrieval entry point (`retrieve`) that dispatches to `bm25_search` / `dense_search` / `rrf` based on the route chosen by `query_classifier.py` |
| `cross_encoder.py` | Reranks a shortlist of chunk IDs with a cross-encoder, scoring `(query, chunk_text)` pairs jointly (`cross_encode`); also owns the `chunk_id -> chunk text` lookup (`get_chunk_text`) used by generation and verification |
| `query_classifier.py` | Query routing: an LLM call classifies a question as `keyword`, `semantic`, or `hybrid` so `retrieval.py` picks the matching strategy |
| `Generation.py` | Answer generation: builds a numbered passage context from the reranked shortlist and asks the generation model to answer with `[n]` citations, abstaining with `NOT_FOUND_IN_CONTRACT` when the passages don't support an answer |
| `verify.py` | Self-verification: a second, independent LLM call checks whether the generated answer is actually grounded in its cited passages; a failed check overrides the answer to abstain, even if generation didn't catch its own mistake |
| `guardrails.py` | Input guardrails run before anything else: a length cap (`MAX_QUERY_CHARS`) and Groq's `llama-prompt-guard-2-86m` classifier to catch prompt-injection/jailbreak attempts; either one tripping short-circuits straight to an abstained response |
| `app.py` | Wires the full pipeline together (`guardrails -> classify -> retrieve -> rerank -> generate -> verify`) via `answer_query()`, and runs an interactive Q&A loop |
| `eval.py` | Ablation harness: pulls CUAD-QA's own gold question/answer pairs for the sample contract, runs each through all three retrieval routes via `answer_query()`, and reports retrieval recall, abstain accuracy, and citation accuracy per route |
| `find_chunk.py` | Dev script: searches the chunks already built by `app.py` for a literal term, to sanity-check what a given query should be able to retrieve |
| `test.py` | Evaluation harness for the query router — runs 10 hand-labeled questions through `classify()` and prints PASS/FAIL per case plus an overall score |

---

## Data Flow (current pipeline)

**One-time setup, at startup (`app.py`):**

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
embed_and_chunk()      (BM25 index is rebuilt fresh per query, see below)
(sentence-transformers)
        │
        ▼
     ChromaDB
(dense vectors, persisted to disk)
```

**Per query (`app.py: answer_query()`):**

```
User query
        │
        ▼
   Input guardrails (guardrails.py)
   length cap + Prompt Guard injection check → abstain if either trips
        │
        ▼
   Query routing (query_classifier.py)
   keyword | semantic | hybrid
        │
        ├── keyword ──→ bm25_search()      → {chunk_id: rank}
        ├── semantic ─→ dense_search()      → {chunk_id: rank}
        └── hybrid ───→ dense_search() + bm25_search()
                         → Reciprocal Rank Fusion (RRF.py)
                           score = 1/(k+rank_dense) + 1/(k+rank_bm25)
        │
        ▼
     Top-10 candidate chunk IDs (retrieval.py)
        │
        ▼
   Cross-encoder reranking (cross_encoder.py)
   scores (query, chunk_text) pairs jointly
        │
        ▼
     Final top-5 ranked chunks
        │
        ▼
   LLM answer generation (Generation.py)
   numbered passages → answer with [n] citations, or NOT_FOUND_IN_CONTRACT
        │
        ▼
   Self-verification (verify.py)
   second LLM call checks the answer is grounded in its cited passages;
   a failed check forces the answer to abstain
```

`eval.py` runs this same per-query pipeline (via `answer_query()`, with `route` forced to each strategy in turn) against CUAD-QA's gold question/answer pairs to measure which retrieval strategy actually performs best.

---

## Architecture Roadmap

| # | Stage | Status |
|---|---|---|
| 1 | Chunking | ✅ Done |
| 2 | Dense retrieval (embeddings + ChromaDB) | ✅ Done |
| 3 | BM25 keyword retrieval | ✅ Done |
| 4 | Hybrid retrieval — dense + BM25 via RRF | ✅ Done |
| 5 | Cross-encoder reranking | ✅ Done |
| 6 | Query routing | ✅ Done |
| 7 | Self-verification pass (LLM-as-judge grounding check) | ✅ Done |
| 8 | Guardrails layer (length cap + prompt-injection classifier) | ✅ Done |
| 9 | Ablation eval harness | ✅ Done |
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
- **Why the query router defaults to `hybrid` instead of failing loudly.** A malformed or unparseable LLM response (or a dropped API call) shouldn't take retrieval down entirely — `hybrid` is the safest fallback since it's a superset of what `keyword` or `semantic` alone would retrieve. The router's own accuracy is checked separately via `test.py`'s 10 hand-labeled cases rather than trusted blindly.
- **Why `verify.py` uses `max_tokens=512` for a one-word YES/NO verdict.** `openai/gpt-oss-20b` (also used for routing) is a reasoning model — it spends part of its token budget on a hidden `reasoning` field before writing the visible answer. With a tight `max_tokens=5`, the budget was consumed entirely by reasoning and the visible `content` came back empty, which parsed as "not grounded" and made every answer abstain regardless of correctness. The fix isn't about verbosity, it's giving the model room to finish reasoning before it outputs the word being parsed.
- **Why `guardrails.py` parses the Prompt Guard response as a float, not a label.** `meta-llama/llama-prompt-guard-2-86m` doesn't reply with a normal chat message — its `content` field is a bare numeric string (e.g. `"0.9996"`), the model's estimated probability that the input is a prompt-injection/jailbreak attempt. Treating it like a normal chat reply and checking `.startswith("YES")` or similar would silently always fail; it has to be parsed as `float(...)` and compared against a threshold instead.
- **Why every module imports `from settings import ...` in lowercase, even though it's easy to typo as `Settings`.** The file is tracked in git as `settings.py`. Windows' filesystem is case-insensitive, so a local rename to `Settings.py` (or a stray `from Settings import ...`) works silently on this machine but would fail with `ModuleNotFoundError` the moment the repo is cloned onto a case-sensitive filesystem — Linux CI, most deployment targets, some macOS setups. Several imports drifted to the capitalized form during development without anyone noticing, precisely because the bug is invisible on Windows.
- **Why `eval.py` groups CUAD-QA rows by question text before scoring.** CUAD repeats the same question text once per clause instance it applies to (e.g. "Parties" appears once per party named in the contract), so the raw dataset has multiple rows sharing one question with different single-span answers. Scoring each row independently would unfairly penalize the system for citing a different — but still valid — gold location than the one a particular row happened to record. Grouping by question and unioning the answer spans into one gold set per unique question fixes that.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install sentence-transformers chromadb rank-bm25 datasets groq python-dotenv nltk
```

> Note: CUAD requires `datasets<4.0.0` for compatibility.
> Query routing, generation, verification, and guardrails all call Groq — set a `GROQ_API_KEY` in a `.env` file at the project root.

---

## Status

Actively in development. This README is updated as each module is completed — see commit history for granular progress.