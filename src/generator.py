from src.logger import setup_logger
from config import COHERE_API_KEY, GROQ_API_KEY, LLM_MODEL, MAX_TOKENS, TEMPERATURE

logger = setup_logger("generator")

def rerank_results(query, results, top_n=3):
    logger.info(f"Reranking {len(results)} results with Cohere")
    import cohere
    co = cohere.ClientV2(api_key=COHERE_API_KEY)
    documents = [r["text"] for r in results]
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents,
        top_n=top_n
    )
    reranked = []
    for item in response.results:
        original = results[item.index]
        reranked.append({
            "text": original["text"],
            "metadata": original["metadata"],
            "rerank_score": round(item.relevance_score, 4),
            "original_index": item.index
        })
    logger.info(f"Reranked to top {len(reranked)} results")
    return reranked

def build_prompt(query, reranked_results):
    context = ""
    for i, result in enumerate(reranked_results):
        source = result["metadata"].get("file_name", "unknown")
        context += f"[Source {i+1}: {source}]\n"
        context += result["text"] + "\n\n"
    prompt = "You are a helpful AI assistant. Answer the question based ONLY on the provided context.\n\n"
    prompt += "Context:\n" + context + "\n"
    prompt += "Question: " + query + "\n\n"
    prompt += "Instructions:\n"
    prompt += "- Answer based only on the context above\n"
    prompt += "- Cite sources using [Source 1], [Source 2] etc\n"
    prompt += "- Be concise and accurate\n\n"
    prompt += "Answer:"
    return prompt

def generate_answer(query, reranked_results):
    logger.info(f"Generating answer for: {query}")
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = build_prompt(query, reranked_results)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )
    answer = response.choices[0].message.content
    logger.info("Answer generated successfully")
    return {
        "answer": answer,
        "sources": [r["metadata"].get("file_name") for r in reranked_results],
        "rerank_scores": [r["rerank_score"] for r in reranked_results]
    }

def rag_pipeline(query, bm25, chunks, vectorstore, model):
    logger.info(f"Running full RAG pipeline for: {query}")
    from src.retriever import hybrid_search
    retrieved = hybrid_search(query, bm25, chunks, vectorstore, model, top_k=5)
    reranked = rerank_results(query, retrieved, top_n=3)
    result = generate_answer(query, reranked)
    logger.info("RAG pipeline complete")
    return result