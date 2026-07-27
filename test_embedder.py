from src.loader import load_document
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks, search

print("=" * 50)
print("Loading model...")
print("=" * 50)
model = load_embedding_model()

print("=" * 50)
print("Loading and chunking document...")
print("=" * 50)
doc = load_document("data/documents/rag_intro.txt")
chunks = chunk_document(doc, strategy="recursive")
print(f"Total chunks: {len(chunks)}")

print("=" * 50)
print("Embedding chunks...")
print("=" * 50)
embedded = embed_chunks(chunks, model)
print(f"Embedded {len(embedded)} chunks")
print(f"Vector size: {len(embedded[0]['vector'])} dimensions")

print("=" * 50)
print("Searching...")
print("=" * 50)
queries = [
    "What is RAG?",
    "How do vector databases work?",
    "What are embeddings?"
]

for query in queries:
    print(f"Query: {query}")
    results = search(query, embedded, model, top_k=2)
    for i, result in enumerate(results):
        print(f"  Result {i+1}: score={result['score']}")
        print(f"  Text: {result['text'][:100]}...")
    print()
