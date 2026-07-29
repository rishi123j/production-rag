from src.loader import load_document
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks
from src.vectorstore import create_client, create_collection, add_chunks
from src.retriever import create_bm25_index
from src.generator import rag_pipeline

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
print("Pipeline ready!")

print("=" * 50)
print("Running full RAG pipeline...")
print("=" * 50)

queries = [
    "What is RAG and how does it work?",
    "What are embeddings?",
    "Why do we need vector databases?"
]

for query in queries:
    print(f"Question: {query}")
    print("-" * 40)
    result = rag_pipeline(query, bm25, chunks, collection, model)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print(f"Rerank scores: {result['rerank_scores']}")
    print()