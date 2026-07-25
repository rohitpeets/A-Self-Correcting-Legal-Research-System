query="can the tenenat terminate the lease early?"
from Chunk import chunk_contract

with open("data/processed/sample_contract.txt","r",encoding="utf-8") as f:
    test_case=f.read()
chunks_f=chunk_contract(test_case,5,1)