from sentence_transformers import SentenceTransformer
from Chunk import chunk_contract
with open("data/processed/sample_contract.txt","r",encoding="utf-8") as f:
    test_case=f.read()
chunks=chunk_contract(test_case,5,1)
model=SentenceTransformer('all-MiniLM-L6-v2')
chunk_vectors=model.encode(chunks)

import chromadb
client=chromadb.PersistentClient(path="data/processed/chroma_db")
collection=client.get_or_create_collection(name="legal_contracts")
embeddings_list=chunk_vectors.tolist()
collection.add(embeddings=embeddings_list,
               documents=chunks,
               ids=[f"chunk_{i}" for i in range (0,chunk_vectors.shape[0])])

test_string="can the tenenat terminate the lease early?"
test_embedding=[1]
test_embedding[0]=model.encode(test_string)

print(collection.query(query_embeddings=test_embedding,n_results=3))