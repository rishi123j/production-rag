# test_loader.py — Test our document loader

from src.loader import load_document, load_all_documents

print("=" * 50)
print("TEST 1: Load single TXT file")
print("=" * 50)

doc = load_document("data/documents/rag_intro.txt")

if doc:
    print(f"File: {doc['metadata']['file_name']}")
    print(f"Type: {doc['metadata']['file_type']}")
    print(f"Characters: {doc['metadata']['char_count']}")
    print(f"First 200 chars:")
    print(doc['text'][:200])
else:
    print("Failed to load document!")

print()
print("=" * 50)
print("TEST 2: Load all documents")
print("=" * 50)

docs = load_all_documents()
print(f"Total documents loaded: {len(docs)}")
for doc in docs:
    print(f"→ {doc['metadata']['file_name']} ({doc['metadata']['char_count']} chars)")
