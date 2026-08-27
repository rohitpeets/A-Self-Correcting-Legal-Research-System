from settings import RERANK_CANDIDATES
from bm25_embedding import bm25_search
from dense_search import dense_search
from RRF import rrf
def retrieve(query,route,model,chunks,collection):
    match route:
        case "keyword":
            return bm25_search(query,chunks,n_results=RERANK_CANDIDATES)
        case "semantic":
            return dense_search(query,model,collection,n_results=RERANK_CANDIDATES)
        case "hybrid":
            return rrf(query,model,chunks,collection,rrf_n=RERANK_CANDIDATES)
        case _:
            raise ValueError(f"unknown route: {route}") 