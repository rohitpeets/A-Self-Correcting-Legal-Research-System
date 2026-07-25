
def clean_token(token):
    import string
    clean=token.lower().translate(str.maketrans('','',string.punctuation)).split()
    return clean

def clean_chunks(input_chunks):
    output=[clean_token(chunk) for chunk in input_chunks]
    return output

def clean_query(input):
    return clean_token(input)

def bm25_search(query,chunks,n_results=5):
    from rank_bm25 import BM25Okapi
    bm25=BM25Okapi(clean_chunks(chunks))
    score_list=bm25.get_scores(clean_query(query))
    result=[]
    for i in range (len(chunks)):
        result.append((score_list[i],f"chunk_{i}"))
    outlist=sorted(result,reverse=True)
    final=outlist[:n_results]
    return final
