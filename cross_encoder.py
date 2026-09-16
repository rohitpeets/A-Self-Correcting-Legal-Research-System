from settings import FINAL_K

def get_chunk_text(id,chunks):
    idx=int(id.split("_")[1])
    return chunks[idx]

def rerank(query,chunks,chunk_ids):
    out_vals=[]
    for i in range(len(chunk_ids)):
        id=chunk_ids[i]
        out_vals.append([query,get_chunk_text(id,chunks)])
    return out_vals

def cross_encode(query, chunks, chunk_ids):
    from sentence_transformers import CrossEncoder
    if not chunk_ids:
        return []
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    scores = cross_encoder.predict(rerank(query, chunks, chunk_ids))
    paired = [(chunk_ids[i], scores[i]) for i in range(len(chunk_ids))]
    return sorted(paired, key=lambda pair: pair[1], reverse=True)[:FINAL_K]
