
def get_chunk_text(id,chunks):
    idx=int(id.split("_")[1])
    return chunks[idx]

def rerank(query,chunks,chunk_ids,k=5):
    out_vals=[]
    for i in range(k):
        id=chunk_ids[i]
        out_vals.append([query,get_chunk_text(id,chunks)])
    return out_vals
def cross_encode(query,chunks,chunk_ids):
    from sentence_transformers import CrossEncoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict(rerank(query,chunks,chunk_ids))
    paired = list(zip(chunk_ids[:5], scores))