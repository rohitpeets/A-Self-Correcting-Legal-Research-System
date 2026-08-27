from sentence_transformers import SentenceTransformer
import chromadb
from Chunk import chunk_contract
from embedding import embed_and_chunk
from query_classifier import classify
from retrieval import retrieve
from cross_encoder import cross_encode, get_chunk_text


with open("data/processed/sample_contract.txt","r",encoding="utf-8") as f:
    contract_text=f.read()

chunks=chunk_contract(contract_text,5,1)
model=SentenceTransformer('all-MiniLM-L6-v2')
client=chromadb.PersistentClient(path="data/processed/chroma_db")
collection=client.get_or_create_collection(name="legal_contracts")

if collection.count()==0:
    embed_and_chunk(chunks,model,collection)


def show_results(results,chunks):
    for i in range(len(results)):
        chunk_id=results[i][0]
        score=results[i][1]
        text=get_chunk_text(chunk_id,chunks)
        preview=text[:150]
        print(f"{i+1}. {chunk_id}  score={float(score):.3f}")
        print(f"   {preview}...")
        print()


def main():
    query=input("Question: ")
    route=classify(query)
    print(f"\nroute: {route}")
    print(f"chunks in collection: {collection.count()}\n")
    ids=retrieve(query,route,model,chunks,collection)
    results=cross_encode(query,chunks,ids)
    show_results(results,chunks)


if __name__=="__main__":
    main()