from src.loader import load_document
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks, embed_text
from src.vectorstore import create_client, create_collection, add_chunks, search_collection, get_collection_stats

print("=" * 50)
print("Setting up ChromaDB...")
print("=" * 50)
client = create_client()
collection = create_collection(client)

print("=" * 50)
print("Loading and processing document...")
print("=" * 50)
model = load_embedding_model()
doc = load_document("data/documents/rag_intro.txt")
chunks = chunk_document(doc, strategy="recursive")
embedded = embed_chunks(chunks, model)
print(f"Chunks ready: {len(embedded)}")

print("=" * 50)
print("Adding to ChromaDB...")
print("=" * 50)
add_chunks(collection, embedded)
stats = get_collection_stats(collection)
print(f"Total in ChromaDB: {stats['total_documents']}")

print("=" * 50)
print("Searching ChromaDB...")
print("=" * 50)
queries = [
    "What is RAG?",
    "How do embeddings work?",
    "What are vector databases?"
]

for query in queries:
    print(f"Query: {query}")
    query_vector = embed_text(query, model)
    results = search_collection(collection, query_vector, top_k=2)
    for i, result in enumerate(results):
        print(f"  Result {i+1}: score={result['score']}")
        print(f"  Source: {result['metadata']['file_name']}")
        print(f"  Text: {result['text'][:80]}...")
    print()
