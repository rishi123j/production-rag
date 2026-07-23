# loader.py — Manual document loader
# Rule: Understand every line before using LangChain

import os
from src.logger import setup_logger
from config import DATA_DIR

logger = setup_logger("loader")

def load_txt(file_path: str) -> dict:
    logger.info(f"Loading TXT file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f"TXT loaded successfully: {len(text)} characters")
        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "file_type": "txt",
                "file_name": os.path.basename(file_path),
                "char_count": len(text)
            }
        }
    except Exception as e:
        logger.error(f"Failed to load TXT: {e}")
        return None

def load_pdf(file_path: str) -> dict:
    logger.info(f"Loading PDF file: {file_path}")
    try:
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = ""
        pages = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
                pages.append({
                    "page_num": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
        logger.info(f"PDF loaded: {len(reader.pages)} pages, {len(text)} characters")
        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "file_type": "pdf",
                "file_name": os.path.basename(file_path),
                "total_pages": len(reader.pages),
                "char_count": len(text),
                "pages": pages
            }
        }
    except Exception as e:
        logger.error(f"Failed to load PDF: {e}")
        return None

def load_docx(file_path: str) -> dict:
    logger.info(f"Loading DOCX file: {file_path}")
    try:
        from docx import Document
        doc = Document(file_path)
        text = ""
        paragraphs = []
        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                text += para.text + "\n"
                paragraphs.append({
                    "paragraph_num": i + 1,
                    "text": para.text
                })
        logger.info(f"DOCX loaded: {len(paragraphs)} paragraphs, {len(text)} characters")
        return {
            "text": text,
            "metadata": {
                "source": file_path,
                "file_type": "docx",
                "file_name": os.path.basename(file_path),
                "total_paragraphs": len(paragraphs),
                "char_count": len(text),
                "paragraphs": paragraphs
            }
        }
    except Exception as e:
        logger.error(f"Failed to load DOCX: {e}")
        return None

def load_document(file_path: str) -> dict:
    logger.info(f"Loading document: {file_path}")

    # Check file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None

    # Get file extension
    ext = os.path.splitext(file_path)[1].lower()

    # Route to correct loader
    if ext == ".txt":
        return load_txt(file_path)
    elif ext == ".pdf":
        return load_pdf(file_path)
    elif ext == ".docx":
        return load_docx(file_path)
    else:
        logger.error(f"Unsupported file type: {ext}")
        return None

def load_all_documents(directory: str = DATA_DIR) -> list:
    logger.info(f"Loading all documents from: {directory}")
    documents = []
    supported = [".txt", ".pdf", ".docx"]

    for file_name in os.listdir(directory):
        ext = os.path.splitext(file_name)[1].lower()
        if ext in supported:
            file_path = os.path.join(directory, file_name)
            doc = load_document(file_path)
            if doc:
                documents.append(doc)

    logger.info(f"Total documents loaded: {len(documents)}")
    return documents
