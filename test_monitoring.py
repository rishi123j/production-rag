from src.loader import load_document
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks
from src.vectorstore import create_client, create_collection, add_chunks
from src.retriever import create_bm25_index
from src.generator import rag_pipeline
from src.monitoring import get_langfuse_client, score_trace

print("=" * 50)
print("Setting up RAG pipeline...")
print("=" * 50)
model = load_embedding_model()
doc = load_document("data/documents/rag_intro.txt")
chunks = chunk_document(doc, strategy="recursive")
embedded = embed_chunks(chunks, model)
client = create_client()
collection = create_collection(client)
add_chunks(collection, embedded)
bm25 = create_bm25_index(chunks)

print("=" * 50)
print("Setting up Langfuse...")
print("=" * 50)
langfuse = get_langfuse_client()
print("Langfuse ready!")

print("=" * 50)
print("Running monitored RAG queries...")
print("=" * 50)

queries = [
    "What is RAG?",
    "How do embeddings work?",
    "What are vector databases?"
]

for query in queries:
    print(f"Query: {query}")
    result = rag_pipeline(query, bm25, chunks, collection, model, langfuse=langfuse)
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Top rerank score: {result['rerank_scores'][0]}")
    if "trace_id" in result:
        score_trace(langfuse, result["trace_id"], "rerank_quality", result["rerank_scores"][0])
        print(f"Trace ID: {result['trace_id']}")
    print()

langfuse.flush()
print("All traces sent to Langfuse!")
print("Check your dashboard at cloud.langfuse.com")