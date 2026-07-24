from embedding import dense_ranks
from bm25_embedding import bm_ranks
def rrf(dense_ranks,bm_ranks):
    dense_list=list(dense_ranks.keys())
    merged=dense_list.copy()
    bm_list=list(bm_ranks.keys())
    for chunk in (bm_list):
        if(chunk not in merged):
            merged.append(chunk)
    rrf_in={}
    for n in merged:
        d_score=dense_ranks.get(n)
        bm_score=bm_ranks.get(n)
        rrf_in[n]=[d_score,bm_score]
    rrf_out={}
    for l in merged:
        vals=rrf_in.get(l)
        score=0
        if(vals[0] is not None):
            score+=(1/(60+vals[0]))
        if(vals[1] is not None):
            score+=(1/(60+vals[1]))
        rrf_out[l]=score
    ranked = sorted(rrf_out.items(), key=lambda x: x[1], reverse=True)
    top_n_ids = [chunk_id for chunk_id, score in ranked[:10]]
    return top_n_ids
fl=rrf(dense_ranks=dense_ranks,bm_ranks=bm_ranks)
print(fl)