from src.logger import setup_logger
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
import time

logger = setup_logger("monitoring")

def get_langfuse_client():
    import os
    os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
    from langfuse import Langfuse
    client = Langfuse()
    logger.info("Langfuse client created")
    return client

def trace_rag_query(langfuse, query, retrieved_chunks, reranked_chunks, answer, rerank_scores, latency_ms):
    logger.info(f"Tracing query: {query}")
    try:
        from langfuse.decorators import langfuse_context
        langfuse_context.configure(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
    except Exception as e:
        logger.warning(f"Could not configure decorator context: {e}")

    trace_id = f"trace_{int(time.time())}"
    try:
        event = langfuse.event(
            name="rag_query",
            input={"query": query},
            output={"answer": answer},
            metadata={
                "total_retrieved": len(retrieved_chunks),
                "total_reranked": len(reranked_chunks),
                "latency_ms": latency_ms,
                "top_rerank_score": rerank_scores[0] if rerank_scores else 0,
                "rerank_scores": rerank_scores
            }
        )
        if hasattr(event, 'trace_id'):
            trace_id = event.trace_id
        logger.info(f"Event logged: latency={latency_ms}ms")
    except Exception as e:
        logger.error(f"Event logging failed: {e}")
        logger.info(f"Metrics logged locally: latency={latency_ms}ms scores={rerank_scores}")

    class TraceResult:
        def __init__(self, tid):
            self.id = tid

    return TraceResult(trace_id)

def score_trace(langfuse, trace_id, score_name, score_value, comment=""):
    try:
        if trace_id:
            langfuse.score(
                trace_id=trace_id,
                name=score_name,
                value=score_value,
                comment=comment
            )
            logger.info(f"Score added: {score_name}={score_value}")
    except Exception as e:
        logger.warning(f"Scoring skipped: {e}")
        logger.info(f"Score logged locally: {score_name}={score_value}")

def log_metrics_locally(query, latency_ms, rerank_scores, answer):
    import json
    import datetime
    metrics = {
        "timestamp": str(datetime.datetime.now()),
        "query": query,
        "latency_ms": latency_ms,
        "top_rerank_score": rerank_scores[0] if rerank_scores else 0,
        "rerank_scores": rerank_scores,
        "answer_length": len(answer)
    }
    with open("logs/metrics.jsonl", "a") as f:
        f.write(json.dumps(metrics) + "\n")
    logger.info(f"Metrics saved locally: latency={latency_ms}ms")
    return metrics