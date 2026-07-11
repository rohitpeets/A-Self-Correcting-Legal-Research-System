from sentence_transformers import SentenceTransformer
from Chunk import chunk_contract
import chromadb
with open("data/processed/sample_contract.txt","r",encoding="utf-8") as f:
    test_case=f.read()
chunks=chunk_contract(test_case,5,1)

model=SentenceTransformer('all-MiniLM-L6-v2')
client=chromadb.PersistentClient(path="data/processed/chroma_db")
collection=client.get_or_create_collection(name="legal_contracts")

def embed_and_chunk(chunks,model,collection):
    chunk_vectors=model.encode(chunks)
    embeddings_list=chunk_vectors.tolist()
    collection.add(embeddings=embeddings_list,
               documents=chunks,
               ids=[f"chunk_{i}" for i in range (0,chunk_vectors.shape[0])])

query="can the tenenat terminate the lease early?"
def dense_search(query,model,collection,n_results=5):
    test_embedding=[model.encode(query)]
    return(collection.query(query_embeddings=test_embedding,n_results=n_results))

embed_and_chunk(chunks,model,collection)
print(dense_search(query,model,collection))