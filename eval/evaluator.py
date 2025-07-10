"""
Evaluation script for the RAG pipeline
"""
import asyncio
import json
import pandas as pd
from tqdm import tqdm
import logging
import os

# Adjust path to import from app and eval
import sys
sys.path.append(os.getcwd())

from app.router import RAGPipeline, RAGRequest
from eval.metrics import RAGEvaluatorMetrics, evaluate_single_query
from app.router import RAGResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineEvaluator:
    def __init__(self, config_path="app/config.yaml", test_queries_path="eval/test_queries.json"):
        self.pipeline = RAGPipeline(config_path)
        self.metrics_calculator = RAGEvaluatorMetrics()
        self.test_queries = self._load_test_queries(test_queries_path)

    def _load_test_queries(self, path: str):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load test queries: {e}")
            return []

    async def run_evaluation(self, models_to_test: list = ["local", "gemini"]):
        """
        Run evaluation for the specified models and save the results.
        """
        all_results = []

        for model_provider in models_to_test:
            logger.info(f"--- Starting evaluation for model: {model_provider} ---")
            
            for query_data in tqdm(self.test_queries, desc=f"Evaluating {model_provider}"):
                request = RAGRequest(
                    query=query_data["query"],
                    provider=model_provider,
                    stream=False
                )
                
                # Process query using the RAG pipeline
                response = await self.pipeline.process_query(request)

                if not isinstance(response, RAGResponse):
                    logger.error(f"Expected RAGResponse, but got {type(response)}. Skipping query.")
                    continue
                
                # Prepare pipeline outputs for metrics calculation
                retrieved_context = " ".join([doc['content'] for doc in response.retrieved_docs])
                
                pipeline_outputs = {
                    "intent": response.intent,
                    "retrieved_context": retrieved_context,
                    "llm_response": response.response,
                    "latency": response.latency,
                    "tokens_used": response.tokens_used
                }
                
                # Calculate metrics for this query
                query_metrics = evaluate_single_query(
                    self.metrics_calculator,
                    query_data,
                    pipeline_outputs
                )
                
                # Combine all info into a single result dictionary
                result = {
                    "model_provider": model_provider,
                    "query": query_data["query"],
                    "true_intent": query_data["intent"],
                    "predicted_intent": response.intent,
                    **query_metrics
                }
                all_results.append(result)

        # Save results
        self._save_results(all_results)
        
        return all_results

    def _save_results(self, results: list):
        """Save detailed and summary results to files."""
        if not results:
            logger.warning("No results to save.")
            return
            
        # Save detailed results to CSV
        df = pd.DataFrame(results)
        df.to_csv("eval/evaluation_results_detailed.csv", index=False)
        logger.info("Detailed evaluation results saved to eval/evaluation_results_detailed.csv")
        
        # Save detailed results to JSON
        with open("eval/evaluation_results_detailed.json", 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Detailed evaluation results saved to eval/evaluation_results_detailed.json")
        
        # Calculate and save summary
        summary = df.groupby('model_provider').agg({
            'intent_accuracy': 'mean',
            'relevance_score': 'mean',
            'context_utilization': 'mean',
            'latency': 'mean',
            'token_cost': 'sum'
        }).reset_index()
        
        summary.to_csv("eval/evaluation_summary.csv", index=False)
        logger.info("Evaluation summary saved to eval/evaluation_summary.csv")
        
        print("\n--- Evaluation Summary ---")
        print(summary)
        print("------------------------")

async def main():
    evaluator = PipelineEvaluator()
    await evaluator.run_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 