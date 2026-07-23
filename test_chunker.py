from src.loader import load_document
from src.chunker import chunk_document

print("=" * 50)
print("TEST 1: Recursive Chunking")
print("=" * 50)

doc = load_document("data/documents/rag_intro.txt")
chunks = chunk_document(doc, strategy="recursive")

print(f"Total chunks: {len(chunks)}")
print()
for chunk in chunks:
    print(f"Chunk {chunk['metadata']['chunk_index']}:")
    print(f"Characters: {chunk['metadata']['char_count']}")
    print(f"Text: {chunk['text'][:100]}...")
    print()

print("=" * 50)
print("TEST 2: Fixed Chunking")
print("=" * 50)

chunks_fixed = chunk_document(doc, strategy="fixed")
print(f"Total chunks: {len(chunks_fixed)}")
for chunk in chunks_fixed:
    print(f"Chunk {chunk['metadata']['chunk_index']}: {chunk['metadata']['char_count']} chars")
