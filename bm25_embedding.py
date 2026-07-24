from parasrc import chunks,query
import string
def clean_token(token):
    clean=token.lower().translate(str.maketrans('','',string.punctuation)).split()
    return clean
def clean_chunks(input_chunks):
    output=[clean_token(chunk) for chunk in input_chunks]
    return output
def clean_query(input):
    return clean_token(input)

from rank_bm25 import BM25Okapi
bm25=BM25Okapi(clean_chunks(chunks))
def bm25_search(query,bm25,chunks,n_results=5):
    score_list=bm25.get_scores(clean_query(query))
    result=[]
    for i in range (len(chunks)):
        result.append((score_list[i],f"chunk_{i}"))
    outlist=sorted(result,reverse=True)
    final=outlist[:n_results]
    return final
result = bm25_search(query, bm25, chunks)
out={}
for i in range(len(result)):
    num=i+1
    out[result[i][1]]=num
bm_ranks=out