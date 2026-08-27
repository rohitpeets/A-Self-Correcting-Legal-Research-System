def embed_and_chunk(chunks,model,collection):
    chunk_vectors=model.encode(chunks)
    embeddings_list=chunk_vectors.tolist()
    collection.add(embeddings=embeddings_list,
               documents=chunks,
               ids=[f"chunk_{i}" for i in range (0,chunk_vectors.shape[0])])
