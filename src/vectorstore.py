import chromadb
from src.logger import setup_logger
from config import TOP_K

logger = setup_logger("vectorstore")

def create_client(persist_dir="data/vectorstore"):
    logger.info(f"Creating ChromaDB client at: {persist_dir}")
    client = chromadb.PersistentClient(path=persist_dir)
    logger.info("ChromaDB client created successfully")
    return client

def create_collection(client, name="rag_documents"):
    logger.info(f"Creating collection: {name}")
    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Collection ready: {collection.count()} documents")
    return collection

def add_chunks(collection, embedded_chunks):
    logger.info(f"Adding {len(embedded_chunks)} chunks to ChromaDB")
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in embedded_chunks:
        ids.append(chunk["metadata"]["chunk_id"])
        embeddings.append(chunk["vector"])
        documents.append(chunk["text"])
        metadatas.append({
            "source": chunk["metadata"]["source"],
            "file_name": chunk["metadata"]["file_name"],
            "file_type": chunk["metadata"]["file_type"],
            "chunk_index": chunk["metadata"]["chunk_index"],
            "total_chunks": chunk["metadata"]["total_chunks"],
            "char_count": chunk["metadata"]["char_count"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    logger.info(f"Added chunks. Total in collection: {collection.count()}")

def search_collection(collection, query_vector, top_k=TOP_K):
    logger.info(f"Searching collection for top {top_k} results")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    formatted = []
    for i in range(len(results["ids"][0])):
        formatted.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": round(1 - results["distances"][0][i], 4)
        })
    logger.info(f"Found {len(formatted)} results")
    return formatted

def get_collection_stats(collection):
    count = collection.count()
    logger.info(f"Collection stats: {count} documents")
    return {"total_documents": count}

def reset_collection(client, name="rag_documents"):
    logger.info(f"Resetting collection: {name}")
    client.delete_collection(name)
    logger.info("Collection deleted")
