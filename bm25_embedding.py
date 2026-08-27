
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
    sorted_list=sorted(result,reverse=True)
    out_list=sorted_list[:n_results]
    final=[]
    for i in range(len(out_list)):
        final.append(out_list[i][1])
    return final
