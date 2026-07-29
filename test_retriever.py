from src.loader import load_document
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks
from src.vectorstore import create_client, create_collection, add_chunks
from src.retriever import create_bm25_index, search_bm25, search_vector, hybrid_search

print("=" * 50)
print("Setting up pipeline...")
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
print("TEST 1: BM25 Search")
print("=" * 50)
query = "RAG retrieval generation"
bm25_results = search_bm25(query, bm25, chunks, top_k=2)
for i, r in enumerate(bm25_results):
    print(f"Result {i+1}: score={r['score']} type={r['retrieval_type']}")
    print(f"Text: {r['text'][:80]}...")
    print()

print("=" * 50)
print("TEST 2: Vector Search")
print("=" * 50)
vector_results = search_vector(query, collection, model, top_k=2)
for i, r in enumerate(vector_results):
    print(f"Result {i+1}: score={r['score']} type={r['retrieval_type']}")
    print(f"Text: {r['text'][:80]}...")
    print()

print("=" * 50)
print("TEST 3: Hybrid Search")
print("=" * 50)
hybrid_results = hybrid_search(query, bm25, chunks, collection, model, top_k=3)
for i, r in enumerate(hybrid_results):
    print(f"Result {i+1}: score={r['score']} type={r['retrieval_type']}")
    print(f"Text: {r['text'][:80]}...")
    print()
