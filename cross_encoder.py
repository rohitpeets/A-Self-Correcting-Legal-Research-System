from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
def get_chunk_text(id,chunks):
    idx=int(id.split("_")[1])
    return chunks[idx]
from parasrc import chunks,query
from RRF import fl
def rerank(query=query,chunks=chunks,chunk_ids=fl,k=5):
    out_vals=[]
    for i in range(k):
        id=chunk_ids[i]
        out_vals.append([query,get_chunk_text(id,chunks)])
    return out_vals
print(rerank())
    