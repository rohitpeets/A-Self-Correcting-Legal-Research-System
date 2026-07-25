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

