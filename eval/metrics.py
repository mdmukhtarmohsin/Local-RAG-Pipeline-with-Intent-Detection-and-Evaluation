"""
Metrics for evaluating the RAG pipeline
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

class RAGEvaluatorMetrics:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def calculate_intent_accuracy(self, predicted_intent: str, true_intent: str) -> float:
        """Calculates intent accuracy."""
        return 1.0 if predicted_intent == true_intent else 0.0

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for a given text."""
        return self.embedding_model.encode(text, convert_to_numpy=True)

    def calculate_relevance_score(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        embedding1 = self.get_embedding(text1).reshape(1, -1)
        embedding2 = self.get_embedding(text2).reshape(1, -1)
        
        return cosine_similarity(embedding1, embedding2)[0][0]

    def calculate_context_utilization(self, response: str, context: str) -> float:
        """
        Calculates how much of the context is utilized in the response.
        A simple implementation based on token overlap.
        """
        if not response or not context:
            return 0.0
        
        response_tokens = set(response.lower().split())
        context_tokens = set(context.lower().split())
        
        if not context_tokens:
            return 0.0
            
        common_tokens = response_tokens.intersection(context_tokens)
        
        return len(common_tokens) / len(context_tokens)

def evaluate_single_query(
    metrics_calculator: RAGEvaluatorMetrics,
    query_data: Dict[str, Any],
    pipeline_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates a single query and returns a dictionary of metrics.
    
    Args:
        metrics_calculator: An instance of RAGEvaluatorMetrics.
        query_data: A dictionary with the test query data (intent, query, etc.).
        pipeline_outputs: A dictionary with the outputs from the RAG pipeline.
    
    Returns:
        A dictionary containing all the calculated metrics.
    """
    
    true_intent = query_data.get("intent")
    predicted_intent = pipeline_outputs.get("intent")
    
    retrieved_context = pipeline_outputs.get("retrieved_context", "")
    expected_context = query_data.get("expected_context", "")
    
    llm_response = pipeline_outputs.get("llm_response", "")
    
    # Calculate metrics
    intent_accuracy = metrics_calculator.calculate_intent_accuracy(predicted_intent, true_intent)
    
    relevance_score = metrics_calculator.calculate_relevance_score(retrieved_context, expected_context)
    
    context_utilization = metrics_calculator.calculate_context_utilization(llm_response, retrieved_context)
    
    return {
        "intent_accuracy": intent_accuracy,
        "relevance_score": relevance_score,
        "context_utilization": context_utilization,
        "latency": pipeline_outputs.get("latency", 0),
        "token_cost": pipeline_outputs.get("tokens_used", 0)
    }

if __name__ == "__main__":
    # Example usage
    metrics_calc = RAGEvaluatorMetrics()
    
    sample_query = {
        "intent": "technical_support",
        "query": "How does pagination work?",
        "expected_context": "The API uses cursor-based pagination."
    }
    
    sample_pipeline_output = {
        "intent": "technical_support",
        "retrieved_context": "Pagination can be implemented using a cursor. The documentation explains how to use the 'next_page' token.",
        "llm_response": "To paginate, you should use the 'next_page' token provided in the API response.",
        "latency": 1.5,
        "tokens_used": 50
    }
    
    evaluation_results = evaluate_single_query(metrics_calc, sample_query, sample_pipeline_output)
    print("Evaluation Results:", evaluation_results) 