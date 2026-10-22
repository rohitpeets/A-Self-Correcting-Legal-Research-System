"""Retrieval-only ablation of the three retrieval routes on many CUAD contracts.

eval.py runs the full pipeline (with LLM calls) on the single sample
contract. This script re-measures only the retrieval side -- keyword (BM25),
semantic (dense), hybrid (RRF) and cross-encoder reranking -- on any number
of contracts from a CUAD SQuAD-format JSON file, with no LLM calls.

It reuses the repo's BM25 tokenizer/search, chunking parameters, RRF formula
(k=60), sizes from settings.py (RETRIEVE_N, RERANK_CANDIDATES, FINAL_K) and
eval.py's gold-chunk matching. Dense search uses an exact cosine top-k over
all-MiniLM-L6-v2's L2-normalized vectors, which ranks the same as the
Chroma collection's default L2 distance, so no vector DB is needed.

Usage:
    python eval_cuad_large.py <squad_json> <out_json> [max_contracts]
        [--embed-model PATH_OR_NAME] [--ce-model PATH_OR_NAME]

Example (official CUAD test split, 102 contracts):
    python eval_cuad_large.py CUAD_v1/test.json data/processed/eval_cuad_test_rows.json
"""
import argparse
import json
import math
import re
import time

import numpy as np

from bm25_embedding import bm25_search
from settings import RETRIEVE_N, RERANK_CANDIDATES, FINAL_K, MAX_QUERY_CHARS

ROUTES = ["keyword", "semantic", "hybrid"]
MIN_GOLD_TEXT_LEN = 5  # same as eval.py
RRF_K = 60  # same as RRF.py


def chunk_contract(text, no_lines=5, overlap=1):
    """Same as Chunk.chunk_contract, without its per-call nltk.download()."""
    import nltk
    cleaned = re.sub(r"\s+", " ", text)
    sentences = nltk.sent_tokenize(cleaned)
    chunks, start = [], 0
    while start < len(sentences):
        chunks.append(" ".join(sentences[start:start + no_lines]))
        start += no_lines - overlap
    return chunks


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def find_gold_chunk_ids(gold_texts, chunks):
    """Same as eval.find_gold_chunk_ids: full span, or its first 60 chars."""
    lowered = [c.lower() for c in chunks]
    ids = set()
    for text in gold_texts:
        needle = normalize_whitespace(text).lower()
        if len(needle) < MIN_GOLD_TEXT_LEN:
            continue
        prefix = needle[:60]
        for i, c in enumerate(lowered):
            if needle in c or prefix in c:
                ids.add(f"chunk_{i}")
    return ids


def dense_search(q_vec, chunk_vecs, n):
    order = np.argsort(-(chunk_vecs @ q_vec), kind="stable")[:n]
    return [f"chunk_{i}" for i in order]


def rrf(dense_ids, bm25_ids, n):
    """Same as RRF.rrf: a missing rank contributes 0, k=60."""
    d = {cid: r + 1 for r, cid in enumerate(dense_ids)}
    b = {cid: r + 1 for r, cid in enumerate(bm25_ids)}
    merged = list(dense_ids) + [c for c in bm25_ids if c not in d]
    scores = {}
    for cid in merged:
        s = 0.0
        if cid in d:
            s += 1 / (RRF_K + d[cid])
        if cid in b:
            s += 1 / (RRF_K + b[cid])
        scores[cid] = s
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:n]]


def hit(ids, gold, k):
    return 1 if set(ids[:k]) & gold else 0


def load_contracts(path):
    """One entry per contract; questions grouped by text with answers unioned
    (as eval.py does). Question text is stripped: the SQuAD-format CUAD files
    end some questions with a non-breaking space that the HF cuad-qa copy
    used by eval.py does not have."""
    data = json.load(open(path, encoding="utf-8"))["data"]
    contracts = []
    for doc in data:
        for para in doc["paragraphs"]:
            by_q = {}
            for qa in para["qas"]:
                by_q.setdefault(qa["question"].strip(), []).extend(a["text"] for a in qa["answers"])
            contracts.append({"title": doc["title"], "context": para["context"], "qs": by_q})
    return contracts


def run(src, max_contracts, embed_model, ce_model):
    from sentence_transformers import SentenceTransformer, CrossEncoder
    embedder = SentenceTransformer(embed_model, device="cpu")
    ce = CrossEncoder(ce_model, device="cpu")

    contracts = load_contracts(src)
    if max_contracts:
        contracts = contracts[:max_contracts]

    rows, skipped, unanswerable = [], 0, 0
    t0 = time.time()
    for ci, c in enumerate(contracts):
        chunks = chunk_contract(c["context"])
        chunk_vecs = embedder.encode(chunks, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False)
        answerable = [(q, t) for q, t in c["qs"].items() if t]
        unanswerable += sum(1 for t in c["qs"].values() if not t)
        if answerable:
            q_vecs = embedder.encode([q for q, _ in answerable], normalize_embeddings=True,
                                     show_progress_bar=False)
        for (question, texts), q_vec in zip(answerable, q_vecs if answerable else []):
            gold = find_gold_chunk_ids(texts, chunks)
            if not gold:
                skipped += 1  # gold not locatable in any chunk; skipped, as in eval.py
                continue
            bm = bm25_search(question, chunks, n_results=RETRIEVE_N)
            de = dense_search(q_vec, chunk_vecs, RETRIEVE_N)
            routes = {
                "keyword": bm[:RERANK_CANDIDATES],
                "semantic": de[:RERANK_CANDIDATES],
                "hybrid": rrf(de, bm, RERANK_CANDIDATES),
            }
            row = {"contract": c["title"], "question": question,
                   "n_chunks": len(chunks), "n_gold": len(gold),
                   # app.py's guardrail would block these before retrieval
                   "over_max_query_chars": len(question) > MAX_QUERY_CHARS}
            # Cross-encoder scores depend only on (question, chunk), so score the
            # union of the routes' candidates once and reuse it per route.
            cand = sorted({i for ids in routes.values() for i in ids})
            ce_scores = dict(zip(cand, ce.predict(
                [[question, chunks[int(i.split("_")[1])]] for i in cand],
                show_progress_bar=False))) if cand else {}
            for name, ids in routes.items():
                row[f"{name}_r10"] = hit(ids, gold, RERANK_CANDIDATES)
                # same as cross_encoder.cross_encode: sort by score, keep FINAL_K
                reranked = sorted(ids, key=lambda i: ce_scores[i], reverse=True)[:FINAL_K]
                row[f"{name}_rr5"] = hit(reranked, gold, FINAL_K)
            rows.append(row)
        print(f"[{ci + 1}/{len(contracts)}] {len(rows)} scored, {time.time() - t0:.0f}s",
              flush=True)
    return rows, skipped, unanswerable, len(contracts)


def cluster_bootstrap(rows, key_fn, iters=2000, seed=0):
    """95% CI of mean(key_fn(row)), resampling whole contracts (questions in
    the same contract are not independent)."""
    by_c = {}
    for r in rows:
        by_c.setdefault(r["contract"], []).append(key_fn(r))
    sums = np.array([sum(v) for v in by_c.values()], dtype=float)
    cnts = np.array([len(v) for v in by_c.values()], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(iters, len(sums)))
    means = np.sort(sums[idx].sum(1) / cnts[idx].sum(1))
    return float(means[int(0.025 * iters)]), float(means[int(0.975 * iters) - 1])


def mcnemar_exact_p(b, c):
    """Two-sided exact McNemar (binomial sign test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(rows, skipped, unanswerable, n_contracts):
    n = len(rows)
    out = {"contracts": n_contracts, "answerable_scored": n,
           "answerable_skipped_gold_unlocatable": skipped,
           "unanswerable_not_scored": unanswerable,
           "scored_over_max_query_chars": sum(r["over_max_query_chars"] for r in rows),
           "ci_method": "95% bootstrap, contracts resampled (2000 iters, seed 0)",
           "metrics": {}}
    for metric, label in [("r10", f"recall@{RERANK_CANDIDATES}"), ("rr5", f"rerank_recall@{FINAL_K}")]:
        m = {}
        for route in ROUTES:
            k = f"{route}_{metric}"
            hits = sum(r[k] for r in rows)
            lo, hi = cluster_bootstrap(rows, lambda r: r[k])
            m[route] = {"hits": hits, "n": n, "value": hits / n if n else None, "ci95": [lo, hi]}
        for other in ["keyword", "semantic"]:
            h, o = f"hybrid_{metric}", f"{other}_{metric}"
            win = sum(1 for r in rows if r[h] and not r[o])
            lose = sum(1 for r in rows if r[o] and not r[h])
            lo, hi = cluster_bootstrap(rows, lambda r: r[h] - r[o])
            m[f"hybrid_vs_{other}"] = {"hybrid_only_hits": win, f"{other}_only_hits": lose,
                                       "diff": (win - lose) / n if n else None,
                                       "diff_ci95": [lo, hi],
                                       "mcnemar_exact_p": mcnemar_exact_p(win, lose)}
        out["metrics"][label] = m
    return out


def print_summary(s):
    print(f"\n{s['contracts']} contracts, {s['answerable_scored']} answerable questions scored "
          f"({s['answerable_skipped_gold_unlocatable']} skipped: gold not locatable; "
          f"{s['unanswerable_not_scored']} unanswerable not scored)")
    for label, m in s["metrics"].items():
        print(f"\n{label}")
        for route in ROUTES:
            r = m[route]
            print(f"  {route:<9} {r['value']:.3f} ({r['hits']}/{r['n']})  "
                  f"95% CI [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
        for other in ["keyword", "semantic"]:
            v = m[f"hybrid_vs_{other}"]
            print(f"  hybrid vs {other}: hybrid-only {v['hybrid_only_hits']}, "
                  f"{other}-only {v[f'{other}_only_hits']}, diff {v['diff']:+.3f} "
                  f"CI [{v['diff_ci95'][0]:+.3f}, {v['diff_ci95'][1]:+.3f}], "
                  f"McNemar p={v['mcnemar_exact_p']:.3g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("squad_json")
    ap.add_argument("out_json")
    ap.add_argument("max_contracts", nargs="?", type=int, default=None)
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--summary-json", default=None, help="also write the summary here")
    args = ap.parse_args()

    rows, skipped, unanswerable, n_contracts = run(
        args.squad_json, args.max_contracts, args.embed_model, args.ce_model)
    summary = summarize(rows, skipped, unanswerable, n_contracts)
    print_summary(summary)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
