from src.logger import setup_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBEDDING_MODEL

logger = setup_logger("rag_chain")

def load_documents_langchain(file_path):
    logger.info(f"Loading document with LangChain: {file_path}")
    from langchain_community.document_loaders import TextLoader
    from langchain_community.document_loaders import PyPDFLoader
    import os
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        logger.error(f"Unsupported file type: {ext}")
        return None
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} documents")
    return docs

def chunk_documents_langchain(docs):
    logger.info(f"Chunking with LangChain RecursiveCharacterTextSplitter")
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def create_vectorstore_langchain(chunks, persist_dir="data/vectorstore_langchain"):
    logger.info("Creating vectorstore with LangChain + ChromaDB")
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"Vectorstore created with {vectorstore._collection.count()} documents")
    return vectorstore

def search_langchain(vectorstore, query, top_k=TOP_K):
    logger.info(f"Searching with LangChain: {query}")
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    formatted = []
    for doc, score in results:
        formatted.append({
            "text": doc.page_content,
            "score": round(float(1 - score), 4),
            "metadata": doc.metadata
        })
    logger.info(f"Found {len(formatted)} results")
    return formatted
