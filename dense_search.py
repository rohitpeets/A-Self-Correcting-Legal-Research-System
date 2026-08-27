
def dense_search(query,model,collection,n_results=5):
    test_embedding=[model.encode(query)]
    result=(collection.query(query_embeddings=test_embedding,n_results=n_results))
    return result['ids'][0]

