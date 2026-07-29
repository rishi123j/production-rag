from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from src.loader import load_document, load_all_documents
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunks
from src.vectorstore import create_client, create_collection, add_chunks, get_collection_stats, reset_collection
from src.retriever import create_bm25_index
from src.generator import rag_pipeline
from src.logger import setup_logger
from config import DATA_DIR

logger = setup_logger("main")

app = FastAPI(
    title="Production RAG API",
    description="Ask questions about your documents",
    version="1.0"
)

model = None
client = None
collection = None
bm25 = None
chunks = []

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list
    rerank_scores: list

@app.on_event("startup")
async def startup():
    global model, client, collection
    logger.info("Starting RAG API...")
    model = load_embedding_model()
    client = create_client()
    collection = create_collection(client)
    logger.info("RAG API ready!")

@app.get("/")
def home():
    return {
        "name": "Production RAG API",
        "version": "1.0",
        "endpoints": ["/health", "/index", "/ask", "/documents", "/reset"]
    }

@app.get("/health")
def health():
    stats = get_collection_stats(collection)
    return {
        "status": "healthy",
        "total_documents": stats["total_documents"],
        "model_loaded": model is not None
    }

@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    global bm25, chunks
    logger.info(f"Indexing file: {file.filename}")
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    doc = load_document(file_path)
    if not doc:
        raise HTTPException(status_code=400, detail="Could not load document")
    new_chunks = chunk_document(doc, strategy="recursive")
    embedded = embed_chunks(new_chunks, model)
    add_chunks(collection, embedded)
    chunks.extend(new_chunks)
    bm25 = create_bm25_index(chunks)
    return {
        "message": f"Successfully indexed {file.filename}",
        "chunks_created": len(new_chunks),
        "total_chunks": len(chunks)
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask(request: QuestionRequest):
    if not chunks:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Use /index first.")
    logger.info(f"Question received: {request.question}")
    result = rag_pipeline(request.question, bm25, chunks, collection, model)
    return QuestionResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        rerank_scores=result["rerank_scores"]
    )

@app.get("/documents")
def list_documents():
    stats = get_collection_stats(collection)
    return {
        "total_chunks": stats["total_documents"],
        "total_documents": len(set([c["metadata"]["file_name"] for c in chunks])) if chunks else 0,
        "documents": list(set([c["metadata"]["file_name"] for c in chunks])) if chunks else []
    }

@app.delete("/reset")
def reset():
    global bm25, chunks
    reset_collection(client)
    chunks = []
    bm25 = None
    return {"message": "All documents cleared successfully"}