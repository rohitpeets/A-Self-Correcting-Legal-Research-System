
from settings import RETRIEVE_N
def to_rank_dict(id_list):
    out={}
    for i in range(len(id_list)):
        num=i+1
        out[id_list[i]]=num
    return out
    
def rrf(query,model,chunks,collection,rrf_n):

    from dense_search import dense_search
    dense_search_list=dense_search(query,model,collection,n_results=RETRIEVE_N)
    dense_ranks=to_rank_dict(dense_search_list)


    from bm25_embedding import bm25_search
    bm25_search_list=bm25_search(query,chunks,n_results=RETRIEVE_N)
    bm_ranks=to_rank_dict(bm25_search_list)


    dense_rrf_list=list(dense_ranks.keys())
    merged=dense_rrf_list.copy()
    bm_rrf_list=list(bm_ranks.keys())
    for chunk in (bm_rrf_list):
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
    top_n_ids = [chunk_id for chunk_id, score in ranked[:rrf_n]]
    return top_n_ids[:rrf_n]
