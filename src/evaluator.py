from src.logger import setup_logger
from config import GROQ_API_KEY, LLM_MODEL

logger = setup_logger("evaluator")

def create_evaluation_dataset(questions, answers, contexts, ground_truths):
    from datasets import Dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    logger.info(f"Created evaluation dataset with {len(questions)} samples")
    return dataset

def evaluate_rag(dataset):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from langchain_groq import ChatGroq
    from langchain_openai import OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_huggingface import HuggingFaceEmbeddings
    logger.info("Running Ragas evaluation...")
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    faithfulness.llm = ragas_llm
    faithfulness.embeddings = ragas_embeddings
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_precision.embeddings = ragas_embeddings
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision]
    )
    logger.info("Ragas evaluation complete!")
    return result

def print_evaluation_results(results):
    print("=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    df = results.to_pandas()
    print("Columns available:", df.columns.tolist())
    print(df.to_string())
    print()
    print("=" * 50)
    print("AVERAGE SCORES")
    print("=" * 50)
    for col in df.columns:
        if col not in ["question", "answer", "contexts", "ground_truth"]:
            try:
                print(f"{col}: {df[col].mean():.4f}")
            except:
                pass
    return df