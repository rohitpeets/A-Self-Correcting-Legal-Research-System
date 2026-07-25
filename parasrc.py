from sentence_transformers import SentenceTransformer
from Chunk import chunk_contract
import chromadb

model=SentenceTransformer('all-MiniLM-L6-v2')
client=chromadb.PersistentClient(path="data/processed/chroma_db")
collection=client.get_or_create_collection(name="legal_contracts")


