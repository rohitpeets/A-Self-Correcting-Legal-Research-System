import json
import random
from collections import defaultdict

from settings import RERANK_CANDIDATES, FINAL_K
from app import chunks, answer_query

ROUTES = ["keyword", "semantic", "hybrid"]
MIN_GOLD_TEXT_LEN = 5
SAMPLE_CONTRACT_PATH = "data/processed/sample_contract.txt"
RESULTS_PATH = "data/processed/eval_results.json"


def load_gold_questions():
    """Pull CUAD-QA's own gold question/answer pairs for the sample contract.

    CUAD repeats the same question text for each separate clause instance it
    applies to, so rows are grouped by question text and their answer spans
    unioned into one gold set per unique question.
    """
    from datasets import load_dataset

    with open(SAMPLE_CONTRACT_PATH, "r", encoding="utf-8") as f:
        sample = f.read()

    dataset = load_dataset("theatticusproject/cuad-qa", trust_remote_code=True)
    matches = [row for row in dataset["train"] if row["context"] == sample]

    by_question = defaultdict(list)
    for row in matches:
        by_question[row["question"]].append(row)

    questions = []
    for question, rows in by_question.items():
        gold_texts = [t for row in rows for t in row["answers"]["text"]]
        questions.append({
            "question": question,
            "answerable": len(gold_texts) > 0,
            "gold_texts": gold_texts,
        })
    return questions


def find_gold_chunk_ids(gold_texts):
    """Chunks whose text contains a gold answer span.

    Matches on the full span, or its first 60 chars for long clause
    extracts that may run past a single chunk's sentence window -- an
    approximation, not exact offset alignment.
    """
    ids = set()
    for text in gold_texts:
        if len(text) < MIN_GOLD_TEXT_LEN:
            continue
        needle = text.lower()
        prefix = needle[:60]
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            if needle in chunk_lower or prefix in chunk_lower:
                ids.add(f"chunk_{i}")
    return ids


def recall_at_k(retrieved_ids, gold_ids, k):
    if not gold_ids:
        return None
    return 1.0 if set(retrieved_ids[:k]) & gold_ids else 0.0


def evaluate_question(q, route):
    gold_ids = find_gold_chunk_ids(q["gold_texts"]) if q["answerable"] else set()
    if q["answerable"] and not gold_ids:
        return None  # gold text not locatable in any chunk; skip rather than mis-score

    result = answer_query(q["question"], route=route)
    retrieved_ids = result["retrieved"]
    reranked_ids = [chunk_id for chunk_id, _score in result["reranked"]]

    abstain_correct = (not result["abstained"]) if q["answerable"] else result["abstained"]

    citation_correct = None
    if q["answerable"] and not result["abstained"]:
        cited_ids = {chunk_id for _n, chunk_id in result["citations"]}
        citation_correct = bool(cited_ids & gold_ids)

    return {
        "question": q["question"],
        "answerable": q["answerable"],
        "route": route,
        "retrieval_recall": recall_at_k(retrieved_ids, gold_ids, RERANK_CANDIDATES),
        "rerank_recall": recall_at_k(reranked_ids, gold_ids, FINAL_K),
        "abstain_correct": abstain_correct,
        "citation_correct": citation_correct,
        "verified": result["verified"],
    }


def summarize(rows):
    def avg(key, keep=lambda r: True):
        vals = [r[key] for r in rows if keep(r) and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n": len(rows),
        f"retrieval_recall@{RERANK_CANDIDATES}": avg("retrieval_recall"),
        f"rerank_recall@{FINAL_K}": avg("rerank_recall"),
        "abstain_accuracy": avg("abstain_correct"),
        "citation_accuracy": avg("citation_correct", lambda r: r["answerable"]),
    }


def run_eval(routes=ROUTES, limit=None, seed=42):
    questions = load_gold_questions()
    random.Random(seed).shuffle(questions)
    if limit:
        questions = questions[:limit]

    rows_by_route = {}
    for route in routes:
        rows = []
        for q in questions:
            row = evaluate_question(q, route)
            if row is not None:
                rows.append(row)
        rows_by_route[route] = rows

    summary = {route: summarize(rows) for route, rows in rows_by_route.items()}
    print_summary(summary)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows_by_route}, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")

    return summary


def print_summary(summary):
    def fmt(v):
        return f"{v:.2f}" if v is not None else "n/a"

    headers = ["route", "n", f"recall@{RERANK_CANDIDATES}", f"recall@{FINAL_K}", "abstain_acc", "citation_acc"]
    print(f"{headers[0]:<10} {headers[1]:>4} {headers[2]:>10} {headers[3]:>10} {headers[4]:>12} {headers[5]:>13}")
    for route, s in summary.items():
        print(
            f"{route:<10} {s['n']:>4} "
            f"{fmt(s[f'retrieval_recall@{RERANK_CANDIDATES}']):>10} "
            f"{fmt(s[f'rerank_recall@{FINAL_K}']):>10} "
            f"{fmt(s['abstain_accuracy']):>12} "
            f"{fmt(s['citation_accuracy']):>13}"
        )


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_eval(limit=limit)
