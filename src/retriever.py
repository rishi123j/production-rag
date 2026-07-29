import numpy as np
from src.logger import setup_logger
from config import TOP_K

logger = setup_logger("retriever")

def create_bm25_index(chunks):
    logger.info(f"Creating BM25 index for {len(chunks)} chunks")
    from rank_bm25 import BM25Okapi
    tokenized = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    logger.info("BM25 index created successfully")
    return bm25

def search_bm25(query, bm25, chunks, top_k=TOP_K):
    logger.info(f"BM25 searching: {query}")
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text": chunks[idx]["text"],
                "score": round(float(scores[idx]), 4),
                "metadata": chunks[idx]["metadata"],
                "retrieval_type": "bm25"
            })
    logger.info(f"BM25 found {len(results)} results")
    return results

def search_vector(query, vectorstore, model, top_k=TOP_K):
    logger.info(f"Vector searching: {query}")
    from src.embedder import embed_text
    query_vector = embed_text(query, model)
    from src.vectorstore import search_collection
    results = search_collection(vectorstore, query_vector, top_k=top_k)
    for r in results:
        r["retrieval_type"] = "vector"
    logger.info(f"Vector search found {len(results)} results")
    return results

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    logger.info("Applying Reciprocal Rank Fusion")
    scores = {}
    texts = {}
    metadatas = {}
    for rank, result in enumerate(bm25_results):
        key = result["text"][:50]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        texts[key] = result["text"]
        metadatas[key] = result["metadata"]
    for rank, result in enumerate(vector_results):
        key = result["text"][:50]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        texts[key] = result["text"]
        metadatas[key] = result["metadata"]
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    results = []
    for key in sorted_keys:
        results.append({
            "text": texts[key],
            "score": round(scores[key], 4),
            "metadata": metadatas[key],
            "retrieval_type": "hybrid"
        })
    logger.info(f"RRF produced {len(results)} results")
    return results

def hybrid_search(query, bm25, chunks, vectorstore, model, top_k=TOP_K):
    logger.info(f"Hybrid search: {query}")
    bm25_results = search_bm25(query, bm25, chunks, top_k=top_k)
    vector_results = search_vector(query, vectorstore, model, top_k=top_k)
    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    top_results = fused[:top_k]
    logger.info(f"Hybrid search returning {len(top_results)} results")
    return top_results
