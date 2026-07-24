from parasrc import chunks,model,collection,query
import nltk
def embed_and_chunk(chunks,model,collection):
    chunk_vectors=model.encode(chunks)
    embeddings_list=chunk_vectors.tolist()
    collection.add(embeddings=embeddings_list,
               documents=chunks,
               ids=[f"chunk_{i}" for i in range (0,chunk_vectors.shape[0])])

def dense_search(query,model,collection,n_results=5):
    test_embedding=[model.encode(query)]
    result=(collection.query(query_embeddings=test_embedding,n_results=n_results))
    return result['ids'][0]
if(collection.count()==0):
    embed_and_chunk(chunks,model,collection)
testl=dense_search(query,model,collection)
def build_rank_dict(id_list):
    rank_dict = {}
    for index, chunk_id in enumerate(id_list):
        rank_dict[chunk_id] = index+1
    return rank_dict
dense_ranks=build_rank_dict(testl)
