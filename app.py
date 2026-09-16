import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from sentence_transformers import SentenceTransformer
import chromadb
from Chunk import chunk_contract
from embedding import embed_and_chunk
from query_classifier import classify
from retrieval import retrieve
from cross_encoder import cross_encode, get_chunk_text
from Generation import generate
from verify import verify
from guardrails import check_query
from settings import NOT_FOUND


with open("data/processed/sample_contract.txt","r",encoding="utf-8") as f:
    contract_text=f.read()

chunks=chunk_contract(contract_text,5,1)
model=SentenceTransformer('all-MiniLM-L6-v2')
client=chromadb.PersistentClient(path="data/processed/chroma_db")
collection=client.get_or_create_collection(name="legal_contracts")

if collection.count()==0:
    embed_and_chunk(chunks,model,collection)


 
def answer_query(query, route=None):
    """Full pipeline: guardrails -> route -> retrieve -> rerank -> generate -> verify.

    'route' can be forced to 'keyword'/'semantic'/'hybrid' to bypass the
    classifier, e.g. for ablation comparisons in eval.py.
    """
    safe, reason = check_query(query)
    if not safe:
        return {
            "answer": NOT_FOUND,
            "citations": [],
            "id_map": {},
            "context": "",
            "abstained": True,
            "verified": None,
            "blocked_reason": reason,
            "route": None,
            "retrieved": [],
            "reranked": [],
        }

    if route is None:
        route = classify(query)
    ids = retrieve(query, route, model, chunks, collection)
    results = cross_encode(query, chunks, ids)
    result = generate(query, results, chunks)

    if result["abstained"]:
        result["verified"] = None
    else:
        result["verified"] = verify(query, result["answer"], result["context"])
        if not result["verified"]:
            result["answer"] = NOT_FOUND
            result["citations"] = []
            result["abstained"] = True

    result["blocked_reason"] = None
    result["route"] = route
    result["retrieved"] = ids
    result["reranked"] = results
    return result
 
 
def show(result):
    print(f"\nroute: {result['route']}  |  retrieved: {len(result['retrieved'])}"
          f"  |  reranked: {len(result['reranked'])}")
    print("-" * 70)
 
    if result["abstained"]:
        print("The contract does not appear to address this.")
        print("-" * 70)
        return
 
    print(result["answer"])
    print("-" * 70)
 
    if result["citations"]:
        print("Sources:")
        for n, chunk_id in result["citations"]:
            text = get_chunk_text(chunk_id, chunks)
            print(f"  [{n}] {chunk_id}: {text[:200].strip()}...")
    else:
        print("WARNING: answer contains no citations.")
    print()
 
 
def main():
    print(f"Loaded {collection.count()} chunks. Ctrl-C or empty line to quit.\n")
    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            break
        show(answer_query(query))
 
 
if __name__ == "__main__":
    main()