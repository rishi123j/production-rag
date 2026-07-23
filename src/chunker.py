import re
from src.logger import setup_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = setup_logger("chunker")

def split_by_paragraph(text):
    paragraphs = text.split("\n\n")
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    logger.info(f"Split into {len(paragraphs)} paragraphs")
    return paragraphs

def split_by_sentence(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_by_word(text):
    words = text.split(" ")
    words = [w for w in words if w]
    return words

def create_chunks_fixed(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    logger.info(f"Fixed chunking: size={chunk_size}, overlap={overlap}")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    logger.info(f"Created {len(chunks)} fixed chunks")
    return chunks

def create_chunks_recursive(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    logger.info(f"Recursive chunking: size={chunk_size}, overlap={overlap}")
    chunks = []

    def recursive_split(text):
        if len(text) <= chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            current = ""
            for para in paragraphs:
                if len(current) + len(para) <= chunk_size:
                    current += para + "\n\n"
                else:
                    if current:
                        chunks.append(current.strip())
                    current = para + "\n\n"
            if current:
                chunks.append(current.strip())
            return
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= chunk_size:
                    current += sent + " "
                else:
                    if current:
                        chunks.append(current.strip())
                    current = sent + " "
            if current:
                chunks.append(current.strip())
            return
        words = text.split(" ")
        current = ""
        for word in words:
            if len(current) + len(word) <= chunk_size:
                current += word + " "
            else:
                if current:
                    chunks.append(current.strip())
                current = word + " "
        if current:
            chunks.append(current.strip())

    recursive_split(text)
    logger.info(f"Created {len(chunks)} recursive chunks")
    return chunks

def add_overlap(chunks, overlap=CHUNK_OVERLAP):
    if len(chunks) <= 1:
        return chunks
    overlapped = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            overlapped.append(prev_end + " " + chunk)
    logger.info(f"Added overlap to {len(overlapped)} chunks")
    return overlapped

def chunk_document(document, strategy="recursive"):
    logger.info(f"Chunking document: {document['metadata']['file_name']}")
    text = document["text"]
    if strategy == "fixed":
        raw_chunks = create_chunks_fixed(text)
    else:
        raw_chunks = create_chunks_recursive(text)
    chunks_with_overlap = add_overlap(raw_chunks)
    final_chunks = []
    for i, chunk_text in enumerate(chunks_with_overlap):
        final_chunks.append({
            "text": chunk_text,
            "metadata": {
                "chunk_id": f"{document['metadata']['file_name']}_chunk_{i+1}",
                "chunk_index": i + 1,
                "total_chunks": len(chunks_with_overlap),
                "source": document["metadata"]["source"],
                "file_name": document["metadata"]["file_name"],
                "file_type": document["metadata"]["file_type"],
                "char_count": len(chunk_text)
            }
        })
    logger.info(f"Final chunks: {len(final_chunks)}")
    return final_chunks
