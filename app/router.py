"""
Main RAG Router - Orchestrates the complete pipeline
Handles intent detection, document retrieval, prompt building, and response generation
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import asyncio

from pydantic import BaseModel

from .llm_wrapper import LLMWrapper, LLMResponse, LLMProvider
from .intent_classifier import IntentClassifier, IntentPrediction
from .retriever import RAGRetriever, RetrievalResult
from .prompt_templates import PromptTemplateManager, AdvancedPromptBuilder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    provider: Optional[str] = None  # local, gemini, auto
    stream: bool = False
    context: Optional[Dict[str, Any]] = None


class RAGResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    retrieved_docs: List[Dict[str, Any]]
    provider: str
    latency: float
    tokens_used: Optional[int] = None
    session_id: Optional[str] = None


class RAGMetrics(BaseModel):
    total_requests: int = 0
    intent_accuracy: float = 0.0
    average_latency: float = 0.0
    average_confidence: float = 0.0
    provider_usage: Dict[str, int] = {}
    error_rate: float = 0.0


class RAGPipeline:
    def __init__(self, config_path: str = "app/config.yaml"):
        """Initialize the complete RAG pipeline"""
        self.config_path = config_path
        
        # Core components
        self.llm_wrapper = LLMWrapper(config_path)
        self.intent_classifier = IntentClassifier(config_path)
        self.retriever = RAGRetriever(config_path)
        self.prompt_manager = PromptTemplateManager()
        self.advanced_prompt_builder = AdvancedPromptBuilder(self.prompt_manager)
        
        # Metrics tracking
        self.metrics = RAGMetrics()
        self.request_history = []
        
        # Session management
        self.sessions = {}
        
        logger.info("RAG Pipeline initialized successfully")
    
    async def process_query(self, request: RAGRequest) -> Union[RAGResponse, AsyncGenerator[str, None]]:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            request: RAG request object
            
        Returns:
            RAGResponse for normal requests or AsyncGenerator for streaming
        """
        start_time = time.time()
        
        try:
            # Step 1: Intent Detection
            logger.info(f"Processing query: {request.query[:100]}...")
            intent_prediction = await self.intent_classifier.predict(request.query)
            
            # Step 2: Document Retrieval
            retrieval_results = await self.retriever.retrieve_documents(
                request.query, 
                intent_prediction.intent
            )
            
            # Step 3: Prompt Building
            if request.context:
                # Use advanced prompt builder with context
                prompt_data = self.advanced_prompt_builder.build_conversation_prompt(
                    request.query,
                    retrieval_results,
                    intent_prediction.intent,
                    conversation_history=request.context.get('history'),
                    user_context=request.context.get('user_profile')
                )
            else:
                # Use basic prompt building
                prompt_data = self.prompt_manager.build_prompt(
                    request.query,
                    retrieval_results,
                    intent_prediction.intent
                )
            
            # Step 4: LLM Generation
            provider = LLMProvider(request.provider) if request.provider else None
            
            if request.stream:
                # Return streaming response
                return self._generate_streaming_response(
                    prompt_data, provider, request, intent_prediction, 
                    retrieval_results, start_time
                )
            else:
                # Generate regular response
                llm_response = await self.llm_wrapper.generate(
                    prompt_data["user_prompt"],
                    provider=provider,
                    stream=False,
                    system_prompt=prompt_data["system_prompt"]
                )
                
                # Build final response
                response = self._build_response(
                    llm_response, intent_prediction, retrieval_results,
                    start_time, request.session_id
                )
                
                # Update metrics
                self._update_metrics(response, intent_prediction)
                
                return response
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            latency = time.time() - start_time
            
            return RAGResponse(
                response=f"I apologize, but I encountered an error processing your request: {str(e)}",
                intent="error",
                confidence=0.0,
                retrieved_docs=[],
                provider="error",
                latency=latency,
                session_id=request.session_id
            )
    
    async def _generate_streaming_response(
        self,
        prompt_data: Dict[str, str],
        provider: Optional[LLMProvider],
        request: RAGRequest,
        intent_prediction: IntentPrediction,
        retrieval_results: List[RetrievalResult],
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        try:
            # Yield metadata first
            metadata = {
                "intent": intent_prediction.intent,
                "confidence": intent_prediction.confidence,
                "retrieved_docs": len(retrieval_results),
                "type": "metadata"
            }
            yield f"data: {metadata}\n\n"
            
            # Stream LLM response
            async for chunk in self.llm_wrapper.generate(
                prompt_data["user_prompt"],
                provider=provider,
                stream=True,
                system_prompt=prompt_data["system_prompt"]
            ):
                yield f"data: {chunk}\n\n"
            
            # Yield final metrics
            latency = time.time() - start_time
            final_metadata = {
                "latency": latency,
                "type": "complete"
            }
            yield f"data: {final_metadata}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: Error: {str(e)}\n\n"
    
    def _build_response(
        self,
        llm_response: LLMResponse,
        intent_prediction: IntentPrediction,
        retrieval_results: List[RetrievalResult],
        start_time: float,
        session_id: Optional[str]
    ) -> RAGResponse:
        """Build final RAG response"""
        
        # Format retrieved documents
        docs = []
        for result in retrieval_results:
            docs.append({
                "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                "source": result.source,
                "score": result.score
            })
        
        return RAGResponse(
            response=llm_response.content,
            intent=intent_prediction.intent,
            confidence=intent_prediction.confidence,
            retrieved_docs=docs,
            provider=llm_response.provider,
            latency=time.time() - start_time,
            tokens_used=llm_response.tokens_used,
            session_id=session_id
        )
    
    def _update_metrics(self, response: RAGResponse, intent_prediction: IntentPrediction):
        """Update pipeline metrics"""
        self.metrics.total_requests += 1
        
        # Update provider usage
        if response.provider not in self.metrics.provider_usage:
            self.metrics.provider_usage[response.provider] = 0
        self.metrics.provider_usage[response.provider] += 1
        
        # Update running averages
        self.metrics.average_latency = (
            (self.metrics.average_latency * (self.metrics.total_requests - 1) + response.latency) 
            / self.metrics.total_requests
        )
        
        self.metrics.average_confidence = (
            (self.metrics.average_confidence * (self.metrics.total_requests - 1) + response.confidence)
            / self.metrics.total_requests
        )
        
        # Store request history (keep last 100)
        self.request_history.append({
            "query": response.response[:100],
            "intent": response.intent,
            "confidence": response.confidence,
            "latency": response.latency,
            "provider": response.provider,
            "timestamp": time.time()
        })
        
        if len(self.request_history) > 100:
            self.request_history.pop(0)
    
    async def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get similar queries from history"""
        # Simple similarity based on intent classification
        try:
            query_intent = await self.intent_classifier.predict(query)
            
            similar = []
            for request in self.request_history:
                if request["intent"] == query_intent.intent:
                    similar.append(request)
            
            # Sort by confidence and return top results
            similar.sort(key=lambda x: x["confidence"], reverse=True)
            return similar[:limit]
            
        except Exception as e:
            logger.error(f"Error finding similar queries: {e}")
            return []
    
    def get_metrics(self) -> RAGMetrics:
        """Get current pipeline metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset pipeline metrics"""
        self.metrics = RAGMetrics()
        self.request_history = []
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "status": "healthy",
            "components": {},
            "timestamp": time.time()
        }
        
        # Check LLM wrapper
        try:
            test_response = await self.llm_wrapper.generate(
                "Test message",
                provider=LLMProvider.AUTO
            )
            health_status["components"]["llm"] = {
                "status": "healthy" if not test_response.error else "error",
                "provider": test_response.provider,
                "latency": test_response.latency
            }
        except Exception as e:
            health_status["components"]["llm"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check intent classifier
        try:
            test_intent = await self.intent_classifier.predict("Test query")
            health_status["components"]["intent_classifier"] = {
                "status": "healthy",
                "method": test_intent.method,
                "confidence": test_intent.confidence
            }
        except Exception as e:
            health_status["components"]["intent_classifier"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Check retriever
        try:
            test_results = await self.retriever.retrieve_documents(
                "Test query", "technical_support", top_k=1
            )
            health_status["components"]["retriever"] = {
                "status": "healthy",
                "results_found": len(test_results)
            }
        except Exception as e:
            health_status["components"]["retriever"] = {
                "status": "error", 
                "error": str(e)
            }
        
        # Update overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if "error" in component_statuses:
            health_status["status"] = "degraded"
        
        return health_status
    
    async def build_knowledge_base(self):
        """Build/rebuild the knowledge base"""
        logger.info("Building knowledge base...")
        
        try:
            # Create sample documents if needed
            from .retriever import create_sample_documents
            create_sample_documents()
            
            # Build vector database
            self.retriever.build_vector_database()
            
            logger.info("Knowledge base build complete")
            return {"status": "success", "message": "Knowledge base built successfully"}
            
        except Exception as e:
            logger.error(f"Knowledge base build failed: {e}")
            return {"status": "error", "message": str(e)}


# Utility functions for easy access
async def quick_query(
    query: str, 
    provider: str = "auto",
    stream: bool = False
) -> Union[RAGResponse, AsyncGenerator[str, None]]:
    """Quick utility function for processing queries"""
    pipeline = RAGPipeline()
    request = RAGRequest(
        query=query,
        provider=provider,
        stream=stream
    )
    return await pipeline.process_query(request)


# Session management functions
class SessionManager:
    def __init__(self):
        self.sessions = {}
    
    def create_session(self, user_id: str) -> str:
        """Create new session for user"""
        import uuid
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "history": [],
            "created_at": time.time(),
            "last_activity": time.time()
        }
        
        return session_id
    
    def add_to_session(self, session_id: str, role: str, content: str):
        """Add message to session history"""
        if session_id in self.sessions:
            self.sessions[session_id]["history"].append({
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
            self.sessions[session_id]["last_activity"] = time.time()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Remove old sessions"""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        expired_sessions = [
            sid for sid, session in self.sessions.items()
            if session["last_activity"] < cutoff_time
        ]
        
        for sid in expired_sessions:
            del self.sessions[sid]
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")


if __name__ == "__main__":
    async def test_rag_pipeline():
        """Test the complete RAG pipeline"""
        pipeline = RAGPipeline()
        
        # Build knowledge base first
        print("Building knowledge base...")
        await pipeline.build_knowledge_base()
        
        # Test queries
        test_queries = [
            "Why is my API key returning 403 errors?",
            "How much does the pro plan cost per month?",
            "Can you add dark mode to the dashboard?",
            "I need help with webhook configuration"
        ]
        
        print("\nTesting RAG Pipeline:")
        print("=" * 50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            request = RAGRequest(query=query, provider="auto")
            response = await pipeline.process_query(request)
            
            if isinstance(response, RAGResponse):
                print(f"Intent: {response.intent} (confidence: {response.confidence:.3f})")
                print(f"Provider: {response.provider}")
                print(f"Latency: {response.latency:.3f}s")
                print(f"Retrieved docs: {len(response.retrieved_docs)}")
                print(f"Response: {response.response[:200]}...")
                print("-" * 40)
        
        # Check metrics
        metrics = pipeline.get_metrics()
        print(f"\nPipeline Metrics:")
        print(f"Total requests: {metrics.total_requests}")
        print(f"Average latency: {metrics.average_latency:.3f}s")
        print(f"Average confidence: {metrics.average_confidence:.3f}")
        print(f"Provider usage: {metrics.provider_usage}")
        
        # Health check
        health = await pipeline.health_check()
        print(f"\nHealth Check: {health['status']}")
        for component, status in health["components"].items():
            print(f"  {component}: {status['status']}")
    
    asyncio.run(test_rag_pipeline()) 