\# Production RAG System



A production-grade \*\*Retrieval Augmented Generation (RAG)\*\* system built from scratch with hybrid retrieval, cross-encoder reranking, LLM generation, observability, and automated evaluation.



\## 🏗️ Architecture

Document (PDF/TXT/DOCX)

↓

Document Loader + Chunker

↓

BGE Embeddings (768 dims)

↓

ChromaDB Vector Store

↓

Hybrid Retrieval (BM25 + Vector + RRF)

↓

Cohere Reranking

↓

Groq LLM Generation (Llama 3.3 70B)

↓

Answer + Citations + Monitoring





\## ✨ Features



\- \*\*Hybrid Retrieval\*\* — BM25 keyword search + vector semantic search combined with Reciprocal Rank Fusion

\- \*\*Cohere Reranking\*\* — Cross-encoder reranking for precision improvement

\- \*\*Citation Enforcement\*\* — Every answer cites exact source documents

\- \*\*Langfuse Monitoring\*\* — Full observability with traces, latency tracking, and quality scores

\- \*\*Ragas Evaluation\*\* — Automated evaluation of faithfulness, answer relevancy, and context precision

\- \*\*FastAPI Deployment\*\* — Production REST API with 5 endpoints

\- \*\*Professional Logging\*\* — Structured logging with timestamps and log levels



\## 📊 Evaluation Results



| Metric | Score |

|--------|-------|

| Faithfulness | 0.4444 |

| Answer Relevancy | 0.9367 |

| Context Precision | 0.6667 |

| Overall | 0.6826 |



\## 🛠️ Tech Stack



| Component | Technology |

|-----------|-----------|

| Framework | LangChain |

| Vector DB | ChromaDB |

| Embeddings | BGE-base-en-v1.5 (HuggingFace) |

| Keyword Search | BM25 (rank-bm25) |

| Reranking | Cohere rerank-v3.5 |

| LLM | Groq (Llama 3.3 70B) |

| Monitoring | Langfuse |

| Evaluation | Ragas |

| API | FastAPI + Uvicorn |



\## 🚀 API Endpoints



| Method | Endpoint | Description |

|--------|---------|-------------|

| GET | `/health` | Health check + document count |

| POST | `/index` | Upload and index documents |

| POST | `/ask` | Ask questions with citations |

| GET | `/documents` | List indexed documents |

| DELETE | `/reset` | Clear all documents |



\## 📁 Project Structure



production-rag/

├── config.py # All settings in one place

├── main.py # FastAPI application

├── src/

│ ├── loader.py # Document loader (TXT, PDF, DOCX)

│ ├── chunker.py # Recursive text chunker

│ ├── embedder.py # BGE embedding generation

│ ├── vectorstore.py # ChromaDB vector store

│ ├── retriever.py # Hybrid BM25 + vector retrieval

│ ├── generator.py # Cohere reranking + LLM generation

│ ├── monitoring.py # Langfuse observability

│ ├── evaluator.py # Ragas evaluation

│ └── logger.py # Professional logging

├── data/documents/ # Your document files

└── logs/ # Application logs





\## ⚙️ Setup



```bash

git clone https://github.com/rishi123j/production-rag

cd production-rag

python -m venv venv

source venv/Scripts/activate  # Windows

pip install -r requirements.txt

```



Add API keys to `.env`:



GROQ\_API\_KEY=your\_key

COHERE\_API\_KEY=your\_key

LANGFUSE\_PUBLIC\_KEY=your\_key

LANGFUSE\_SECRET\_KEY=your\_key





Run the API:

```bash

uvicorn main:app --reload

```



\## 🔬 Experiment Tracking



| Experiment | Chunk Size | Top-K | Embedding | Faithfulness | Relevancy |

|-----------|-----------|-------|-----------|-------------|----------|

| Exp 1 | 500 | 3 | BGE-base | 0.4444 | 0.9367 |

| Exp 2 | 300 | 5 | BGE-base | TBD | TBD |

| Exp 3 | 500 | 5 | BGE-large | TBD | TBD |



\## 🎯 What I Learned Building This



\- Built RAG pipeline manually before using LangChain — understood every abstraction

\- Hybrid retrieval (BM25 + Vector) outperforms pure vector search

\- Reranking significantly improves precision over initial retrieval

\- Faithfulness score reveals LLM hallucination patterns

\- Production observability is as important as the model itself



\## 👤 Author



\*\*Rishi\*\* — M.Tech AI \& Data Science

Target: LLM/GenAI Engineer Role

GitHub: github.com/rishi123j





