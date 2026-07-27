from src.rag_chain import load_documents_langchain, chunk_documents_langchain, create_vectorstore_langchain, search_langchain

print("=" * 50)
print("TEST: LangChain Pipeline")
print("=" * 50)

print("Step 1: Loading document...")
docs = load_documents_langchain("data/documents/rag_intro.txt")
print(f"Loaded: {len(docs)} document")

print("Step 2: Chunking...")
chunks = chunk_documents_langchain(docs)
print(f"Chunks: {len(chunks)}")

print("Step 3: Creating vectorstore...")
vectorstore = create_vectorstore_langchain(chunks)
print("Vectorstore ready!")

print("Step 4: Searching...")
queries = [
    "What is RAG?",
    "How do embeddings work?",
    "What are vector databases?"
]

for query in queries:
    print(f"Query: {query}")
    results = search_langchain(vectorstore, query, top_k=2)
    for i, result in enumerate(results):
        print(f"  Result {i+1}: score={result['score']}")
        print(f"  Text: {result['text'][:80]}...")
    print()

print("=" * 50)
print("Comparing Manual vs LangChain")
print("=" * 50)
print("Manual pipeline:   load → chunk → embed → store → search")
print("LangChain pipeline: same steps, less code, more features")
print("Understanding: YOU have both!")
