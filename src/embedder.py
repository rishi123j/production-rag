import numpy as np
from src.logger import setup_logger
from config import EMBEDDING_MODEL

logger = setup_logger("embedder")

def load_embedding_model():
    logger.info("Loading embedding model")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Model loaded successfully")
    return model

def embed_text(text, model):
    embedding = model.encode(text)
    return embedding.tolist()

def embed_chunks(chunks, model):
    logger.info(f"Embedding {len(chunks)} chunks")
    embedded = []
    for chunk in chunks:
        vector = embed_text(chunk["text"], model)
        embedded.append({
            "text": chunk["text"],
            "vector": vector,
            "metadata": chunk["metadata"]
        })
    logger.info(f"Embedded {len(embedded)} chunks")
    return embedded

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot / (mag1 * mag2)

def search(query, embedded_chunks, model, top_k=3):
    logger.info(f"Searching: {query}")
    query_vector = embed_text(query, model)
    results = []
    for chunk in embedded_chunks:
        score = cosine_similarity(query_vector, chunk["vector"])
        results.append({
            "text": chunk["text"],
            "score": round(float(score), 4),
            "metadata": chunk["metadata"]
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]
    logger.info(f"Top {top_k} results found")
    return top_results
