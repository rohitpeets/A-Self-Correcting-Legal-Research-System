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

_cross_encoder = None

def _get_cross_encoder():
    """Load the CrossEncoder once and reuse it.

    Re-instantiating CrossEncoder() on every call re-hits the HuggingFace
    Hub for model metadata even when the weights are already cached locally
    -- slow across many calls, and a single dropped connection kills
    whatever's calling it (e.g. eval.py's ablation run).
    """
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def cross_encode(query, chunks, chunk_ids):
    if not chunk_ids:
        return []
    cross_encoder = _get_cross_encoder()
    scores = cross_encoder.predict(rerank(query, chunks, chunk_ids))
    paired = [(chunk_ids[i], scores[i]) for i in range(len(chunk_ids))]
    return sorted(paired, key=lambda pair: pair[1], reverse=True)[:FINAL_K]
