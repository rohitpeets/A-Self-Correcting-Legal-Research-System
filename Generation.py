import os
import re
from dotenv import load_dotenv
from groq import Groq

from settings import (
    GEN_MODEL,
    GEN_TEMPERATURE,
    GEN_MAX_TOKENS,
    MAX_CHUNK_CHARS,
    NOT_FOUND,
)
from cross_encoder import get_chunk_text

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SYSTEM_PROMPT = f"""You are a legal research assistant answering questions about a
single commercial contract. You are given numbered passages extracted from that contract.

Rules:
1. Answer ONLY from the numbered passages. Never use outside legal knowledge,
and never infer terms that are not written in the passages.

2. Cite every factual claim with its passage number in square brackets,
e.g. [2]. A claim supported by more than one passage cites each: [1] [3].

3. If the passages do not contain the answer, reply with exactly {NOT_FOUND} and
nothing else. Do not apologize or explain.

4. Reproduce the contract's own wording for defined terms, dollar amounts, dates,
durations and notice periods rather than paraphrasing them.

5. Be concise: two to four sentences, unless the question requires enumerating a list of items."""

CITATION_RE = re.compile(r"\[(\d+)\]")

def build_context(results, chunks):
    """Turn reranked (chunk_id, score) pairs into a numbered prompt block.

    Returns (context_string, id_map) where id_map is {citation_number: chunk_id}.
    """
    blocks = []
    id_map = {}
    for i, (chunk_id, _score) in enumerate(results):
        n = i + 1
        id_map[n] = chunk_id
        text = get_chunk_text(chunk_id, chunks)[:MAX_CHUNK_CHARS]

        blocks.append(f"[{n}] {text}")
    return "\n\n".join(blocks), id_map

def extract_citations(answer, id_map):
    """Pull [n] markers out of the answer, in order, deduplicated.

    Returns a list of (citation_number, chunk_id). Numbers the model invented that aren't in id_map are dropped.
    """
    seen = []
    for match in CITATION_RE.findall(answer):
        n = int(match)
        if n in id_map and n not in seen:
            seen.append(n)
    return [(n, id_map[n]) for n in seen]

def generate(query, results, chunks):
    """Answer 'query' from the reranked 'results'.

    Returns {"answer", "citations", "id_map", "context", "abstained"}.
    """
    if not results:
        return {
            "answer": NOT_FOUND,
            "citations": [],
            "id_map": {},
            "context": "",
            "abstained": True,
        }
    context, id_map = build_context(results, chunks)
    user_prompt = f"Passages:\n\n{context}\n\nQuestion: {query}"

    try:
        response = _client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
        )
        answer = (response.choices[0].message.content or "").strip()
    except Exception as e:
        print("GENERATION FAILED:", repr(e))
        return {
            "answer": NOT_FOUND,
            "citations": [],
            "id_map": id_map,
            "context": context,
            "abstained": True,
        }
    abstained = answer.strip().upper().startswith(NOT_FOUND)
    citations = [] if abstained else extract_citations(answer, id_map)

    return {
        "answer": answer,
        "citations": citations,
        "id_map": id_map,
        "context": context,
        "abstained": abstained,
    }
