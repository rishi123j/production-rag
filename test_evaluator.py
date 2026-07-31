from src.loader import load_document
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks
from src.vectorstore import create_client, create_collection, add_chunks
from src.retriever import create_bm25_index
from src.generator import rag_pipeline
from src.evaluator import create_evaluation_dataset, evaluate_rag, print_evaluation_results

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
print("Running RAG on test questions...")
print("=" * 50)

test_questions = [
    "What is RAG?",
    "What are embeddings?",
    "What is a vector database?"
]

ground_truths = [
    "RAG stands for Retrieval Augmented Generation combining retrieval with generation.",
    "Embeddings are numerical representations of text converted into vectors.",
    "Vector databases store embeddings and use ANN algorithms for fast search."
]

questions = []
answers = []
contexts = []

for question in test_questions:
    print(f"Q: {question}")
    result = rag_pipeline(question, bm25, chunks, collection, model)
    questions.append(question)
    answers.append(result["answer"])
    contexts.append([chunks[i]["text"] for i in range(min(3, len(chunks)))])
    print(f"A: {result['answer'][:80]}...")
    print()

print("=" * 50)
print("Running Ragas evaluation...")
print("=" * 50)
dataset = create_evaluation_dataset(questions, answers, contexts, ground_truths)
results = evaluate_rag(dataset)
print_evaluation_results(results)