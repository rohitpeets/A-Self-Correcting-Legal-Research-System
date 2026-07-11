from sentence_transformers import SentenceTransformer
from Chunk import chunk_contract
import chromadb
with open("data/processed/sample_contract.txt","r",encoding="utf-8") as f:
    test_case=f.read()
chunks=chunk_contract(test_case,5,1)

model=SentenceTransformer('all-MiniLM-L6-v2')
client=chromadb.PersistentClient(path="data/processed/chroma_db")
collection=client.get_or_create_collection(name="legal_contracts")
query="can the tenenat terminate the lease early?"
