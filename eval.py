import json
import random
import re
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


def normalize_whitespace(text):
    """Collapse whitespace runs the same way chunk_contract() does.

    CUAD's gold answer spans keep the source PDF's raw spacing (multiple
    spaces, stray newlines), but chunks are built from whitespace-collapsed
    text -- without this, an identical passage never matches as a substring.
    """
    return re.sub(r"\s+", " ", text).strip()


def find_gold_chunk_ids(gold_texts):
    """Chunks whose text contains a gold answer span.

    Matches on the full span, or its first 60 chars for long clause
    extracts that may run past a single chunk's sentence window -- an
    approximation, not exact offset alignment.
    """
    ids = set()
    for text in gold_texts:
        needle = normalize_whitespace(text).lower()
        if len(needle) < MIN_GOLD_TEXT_LEN:
            continue
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


def evaluate_question(q, gold_ids, route):
    try:
        result = answer_query(q["question"], route=route)
    except Exception as e:
        print(f"EVAL ROW FAILED ({route}): {q['question'][:60]!r}: {e!r}")
        return {
            "question": q["question"],
            "answerable": q["answerable"],
            "route": route,
            "retrieval_recall": None,
            "rerank_recall": None,
            "abstain_correct": None,
            "citation_correct": None,
            "verified": None,
            "error": repr(e),
        }

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
        "error": None,
    }


def summarize(rows):
    def avg_and_n(key, keep=lambda r: True):
        vals = [r[key] for r in rows if keep(r) and r[key] is not None]
        return (sum(vals) / len(vals) if vals else None), len(vals)

    retrieval_recall, retrieval_n = avg_and_n("retrieval_recall")
    rerank_recall, rerank_n = avg_and_n("rerank_recall")
    abstain_acc, abstain_n = avg_and_n("abstain_correct")
    citation_acc, citation_n = avg_and_n("citation_correct", lambda r: r["answerable"])

    return {
        "n": len(rows),
        f"retrieval_recall@{RERANK_CANDIDATES}": retrieval_recall,
        f"retrieval_recall@{RERANK_CANDIDATES}_n": retrieval_n,
        f"rerank_recall@{FINAL_K}": rerank_recall,
        f"rerank_recall@{FINAL_K}_n": rerank_n,
        "abstain_accuracy": abstain_acc,
        "abstain_accuracy_n": abstain_n,
        "citation_accuracy": citation_acc,
        "citation_accuracy_n": citation_n,
    }


def run_eval(routes=ROUTES, limit=None, seed=42):
    questions = load_gold_questions()
    random.Random(seed).shuffle(questions)
    if limit:
        questions = questions[:limit]

    usable = []
    skipped = 0
    for q in questions:
        gold_ids = find_gold_chunk_ids(q["gold_texts"]) if q["answerable"] else set()
        if q["answerable"] and not gold_ids:
            skipped += 1  # gold text not locatable in any chunk; skip rather than mis-score
            continue
        usable.append((q, gold_ids))

    rows_by_route = {}
    for route in routes:
        rows_by_route[route] = [evaluate_question(q, gold_ids, route) for q, gold_ids in usable]

    summary = {route: summarize(rows) for route, rows in rows_by_route.items()}
    print_summary(summary, total_questions=len(questions), skipped=skipped)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "skipped_unlocatable": skipped, "rows": rows_by_route}, f, indent=2)
    print(f"\nFull results written to {RESULTS_PATH}")

    return summary


def print_summary(summary, total_questions, skipped):
    def fmt(v):
        return f"{v:.2f}" if v is not None else "n/a"

    print(
        f"{total_questions} unique gold questions, {skipped} answerable ones skipped "
        f"(gold text not locatable in any single chunk) -> {total_questions - skipped} scored per route\n"
    )
    headers = ["route", "recall@10 (n)", "recall@5 (n)", "abstain_acc (n)", "citation_acc (n)"]
    print(f"{headers[0]:<10} {headers[1]:>16} {headers[2]:>16} {headers[3]:>18} {headers[4]:>18}")
    for route, s in summary.items():
        recall10 = f"{fmt(s[f'retrieval_recall@{RERANK_CANDIDATES}'])} ({s[f'retrieval_recall@{RERANK_CANDIDATES}_n']})"
        recall5 = f"{fmt(s[f'rerank_recall@{FINAL_K}'])} ({s[f'rerank_recall@{FINAL_K}_n']})"
        abstain = f"{fmt(s['abstain_accuracy'])} ({s['abstain_accuracy_n']})"
        citation = f"{fmt(s['citation_accuracy'])} ({s['citation_accuracy_n']})"
        print(f"{route:<10} {recall10:>16} {recall5:>16} {abstain:>18} {citation:>18}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_eval(limit=limit)
