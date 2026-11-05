# A Self-Correcting Legal Research System

A RAG system for question answering over legal contracts in CUAD. It combines dense and BM25 retrieval with Reciprocal Rank Fusion, reranks with a cross-encoder, routes each query to a retrieval strategy, and checks its own answers before returning them. An ablation study measures which of those components actually earn their complexity.

Most RAG tutorials stop at "embed chunks, run a similarity search, generate an answer." This project adds hybrid retrieval, reranking, citations, a grounding check and input guardrails, so the system abstains instead of answering confidently when the contract doesn't support an answer.

## Pipeline

```
User query
   │
   ▼
Input guardrails (guardrails.py)
length cap + Prompt Guard injection check → abstain if either trips
   │
   ▼
Query routing (query_classifier.py): keyword | semantic | hybrid
   │
   ├── keyword ──→ bm25_search()
   ├── semantic ─→ dense_search()
   └── hybrid ───→ dense_search() + bm25_search()
                    → RRF: score = 1/(k+rank_dense) + 1/(k+rank_bm25)
   │
   ▼
Top-10 candidates (retrieval.py)
   │
   ▼
Cross-encoder reranking (cross_encoder.py) → top 5
   │
   ▼
Generation (Generation.py)
numbered passages → answer with [n] citations, or NOT_FOUND_IN_CONTRACT
   │
   ▼
Self-verification (verify.py)
a second LLM call checks the answer is grounded in its cited passages;
a failed check forces an abstain
```

At startup, `app.py` chunks the contract and embeds the chunks into a persistent ChromaDB collection. The BM25 index is rebuilt per query.

## Results

### Retrieval on the full CUAD test split (102 contracts, 1,242 questions)

`python eval_cuad_large.py CUAD_v1/test.json <out.json>` runs on CUAD's official test split, which was never used during development (file sha256 `007b6a40…a1a073`). It is retrieval only, with no LLM calls: every answerable question goes through all three routes using the live pipeline's chunking, BM25 tokenizer, RRF (k=60) and sizes, and the cross-encoder then reranks each route's top 10 to a top 5. 1,242 of 1,244 answerable questions were scored; 2 were skipped because their gold span couldn't be found in any chunk.

| route | recall@10 [95% CI] | recall@5 after rerank [95% CI] |
|---|---|---|
| keyword | 0.732 (909/1242) [0.699, 0.767] | 0.626 (778/1242) [0.592, 0.661] |
| semantic | 0.744 (924/1242) [0.713, 0.777] | 0.634 (787/1242) [0.598, 0.669] |
| hybrid | **0.773** (960/1242) [0.747, 0.800] | 0.637 (791/1242) [0.601, 0.672] |

- Hybrid retrieval wins by a significant margin. On the same questions, hybrid finds the gold passage where keyword misses it 107 times versus 56 the other way (+4.1 pts, 95% CI [+1.4, +7.0], McNemar p < 0.001), and beats semantic 84 to 48 (+2.9 pts, CI [+0.8, +4.9], p = 0.002).
- After reranking to the top 5, all three routes land within 1 point of each other and no difference is significant (p = 0.34 and 0.77). Hybrid's extra candidates only help if they survive reranking, so the reranker (or `FINAL_K`) is the next thing to tune.
- Abstain and citation accuracy need three Groq calls per question and weren't re-measured at this scale; they come from the pilot below.
- Summary: `data/processed/eval_cuad_test_summary.json` (rerun the script for per-question rows).

### Single-contract pilot (41 questions)

`python eval.py` runs all 41 unique CUAD-QA gold questions for the sample contract through each route in turn, bypassing the classifier so the routes are compared on identical questions.

| route | recall@10 | recall@5 (post-rerank) | abstain accuracy | citation accuracy |
|---|---|---|---|---|
| keyword | 0.68 | 0.53 | 0.68 | 0.56 (n=9) |
| semantic | 0.74 | 0.63 | 0.68 | 0.43 (n=7) |
| hybrid | 0.79 | 0.53 | 0.73 | 0.20 (n=10) |

- Hybrid retrieves the gold passage most often and has the best abstain accuracy, but the lowest citation accuracy. Retrieving the right chunk more often didn't make generation cite it correctly more often.
- Treat these as directional. Citation accuracy only counts answerable questions the system didn't abstain on (7 to 10 per route), and some transient Groq errors during the run were turned into abstains by the fail-closed error handling, which shrank those samples further.
- Per-question output: `data/processed/eval_results.json`.

## Modules

| File | Purpose |
|---|---|
| `dataset_handler.py` | Pulls a sample contract from CUAD and caches it in `data/processed/sample_contract.txt` |
| `Chunk.py` | Sentence-based chunking (`chunk_contract`) with configurable size and overlap |
| `settings.py` | Shared config: `RETRIEVE_N`, `RERANK_CANDIDATES`, `FINAL_K`, model names, and the `NOT_FOUND` sentinel |
| `embedding.py` | Embeds chunks into ChromaDB (`embed_and_chunk`) |
| `dense_search.py` | Returns ChromaDB's nearest chunk IDs by cosine similarity |
| `bm25_embedding.py` | BM25 tokenization helpers (`clean_token`, `clean_chunks`, `clean_query`) and `bm25_search` |
| `RRF.py` | Fuses dense and BM25 rankings with Reciprocal Rank Fusion (k=60) |
| `retrieval.py` | Single entry point (`retrieve`) that dispatches on the route |
| `cross_encoder.py` | Reranks `(query, chunk_text)` pairs; also owns the chunk ID to text lookup (`get_chunk_text`) |
| `query_classifier.py` | LLM call that labels a question `keyword`, `semantic` or `hybrid` |
| `Generation.py` | Builds numbered passages and generates an answer with `[n]` citations, or abstains |
| `verify.py` | Independent LLM check that the answer is grounded in its cited passages |
| `guardrails.py` | Length cap (`MAX_QUERY_CHARS`) and `llama-prompt-guard-2-86m` injection check |
| `app.py` | Wires the pipeline together in `answer_query()` and runs an interactive Q&A loop |
| `eval.py` | Single-contract ablation: recall, abstain accuracy and citation accuracy per route |
| `eval_cuad_large.py` | Retrieval-only ablation on any CUAD SQuAD-format file, with contract-level bootstrap CIs and paired McNemar tests |
| `find_chunk.py` | Dev script that searches built chunks for a literal term |
| `test.py` | Runs 10 hand-labeled questions through the router and prints PASS/FAIL |

## Stack

- Python, `sentence-transformers` (`all-MiniLM-L6-v2` embeddings, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker), ChromaDB (`PersistentClient`), `rank_bm25` (`BM25Okapi`)
- Groq via the `groq` SDK: `openai/gpt-oss-20b` for routing and verification, `openai/gpt-oss-120b` for generation, `meta-llama/llama-prompt-guard-2-86m` for input screening
- `python-dotenv` for the API key
- Dataset: `theatticusproject/cuad-qa` on HuggingFace (needs `datasets<4.0.0` and `trust_remote_code=True`)

## Decisions and bugs worth knowing about

- **Rank dicts are `{chunk_id: rank}`.** The first version used `{rank: chunk_id}`, which prints nicely but makes "what rank did this chunk get?" an O(n) scan. RRF asks that question constantly, so the key is the chunk ID.
- **A missing rank contributes 0 in RRF.** An early version turned a missing rank into rank 0, which gives `1/(k+0)`, the highest possible score. A chunk only one retriever found should score lower than one both agree on, so the term is now skipped.
- **RRF scores the union of both result lists**, since dense and BM25 don't return the same chunks.
- **ChromaDB has a `collection.count() == 0` guard.** Without it, re-running would add the same IDs again and either error or duplicate vectors. The tradeoff: if the source contract changes, the `chroma_db` folder has to be cleared by hand.
- **The cross-encoder only sees a 10-item shortlist.** It scores each `(query, chunk)` pair in its own forward pass, which is too slow for the whole corpus on every query.
- **BM25 has its own tokenizer.** Unstripped punctuation (`"Company,"` vs `"Company"`) was silently costing keyword matches, and fixing it changed the top-5 BM25 results.
- **The router falls back to `hybrid`** on a malformed response or dropped call, since hybrid covers what either single strategy would retrieve. Router accuracy is checked separately with `test.py`.
- **`verify.py` uses `max_tokens=512` for a YES/NO answer.** `gpt-oss-20b` is a reasoning model and spends tokens on hidden reasoning first. At `max_tokens=5` the visible answer came back empty, which parsed as "not grounded" and made every answer abstain.
- **Prompt Guard's reply is parsed as a float.** Its `content` is a bare probability string like `"0.9996"`, so a label check like `.startswith("YES")` would always fail.
- **Imports use lowercase `settings`.** Windows is case-insensitive, so `from Settings import ...` worked locally but would fail on Linux CI or any case-sensitive filesystem. Several imports had drifted to the capitalized form unnoticed.
- **The `CrossEncoder` instance is cached.** Constructing it per call hit the HuggingFace Hub every time, and one dropped connection near the end of a ~20-minute eval run (`httpx.ReadError`) lost the whole run.
- **`eval.py` groups CUAD-QA rows by question.** CUAD repeats a question once per clause instance with different gold spans, so scoring rows independently would penalize the system for citing a valid location that a particular row didn't record.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install sentence-transformers chromadb rank-bm25 datasets groq python-dotenv nltk
```

CUAD needs `datasets<4.0.0`. Routing, generation, verification and guardrails all call Groq, so put a `GROQ_API_KEY` in a `.env` file at the project root.

## Status

All nine planned stages are done: chunking, dense retrieval, BM25, hybrid RRF, reranking, query routing, self-verification, guardrails, and the ablation harness. Dynamic document ingestion and deployment are stretch goals.
