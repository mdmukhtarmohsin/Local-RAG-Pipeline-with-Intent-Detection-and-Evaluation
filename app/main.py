"""
FastAPI server for the RAG Support Assistant
"""

import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import yaml
from typing import AsyncGenerator

from app.llm_wrapper import LLMWrapper, LLMProvider, LLMResponse as LLMWrapperResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
with open("app/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=config.get('ui', {}).get('title', "Customer Support RAG Assistant"),
    description="A RAG-based customer support assistant with intent detection.",
    version="1.0.0"
)

# Initialize LLM Wrapper
llm_wrapper = LLMWrapper(config_path="app/config.yaml")

class GenerationRequest(BaseModel):
    query: str
    provider: LLMProvider = LLMProvider.AUTO
    stream: bool = False

class GenerationResponse(BaseModel):
    content: str
    provider: str
    model: str
    tokens_used: int | None
    latency: float
    error: str | None

@app.on_event("startup")
async def startup_event():
    # Check Ollama connection on startup
    ollama_available = await llm_wrapper.check_ollama_availability()
    if not ollama_available:
        logger.warning("Ollama is not available. The system will rely on Gemini.")
    else:
        logger.info("Ollama is available.")

@app.get("/")
async def read_root():
    """Root endpoint to check if the server is running."""
    return {"status": "ok", "message": "Welcome to the RAG Support Assistant API!"}

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    Generate a response to a user query.
    This endpoint can either return a complete response or a streaming response.
    """
    if request.stream:
        # This part should return a StreamingResponse
        # but the response_model is not compatible with it.
        # We will handle this in the UI part.
        # For now, let's just return an error if streaming is requested on this endpoint.
        raise HTTPException(status_code=400, detail="Streaming is not supported on this endpoint. Use /generate_stream instead.")

    logger.info(f"Received query: '{request.query}' with provider: {request.provider}")
    
    response = await llm_wrapper.generate(
        prompt=request.query,
        provider=request.provider,
        stream=False
    )
    
    if isinstance(response, LLMWrapperResponse):
        if response.error:
            raise HTTPException(status_code=500, detail=response.error)
        return GenerationResponse(**response.model_dump())
    
    # This should not happen if stream is False
    raise HTTPException(status_code=500, detail="Unexpected response type from LLM wrapper.")

@app.post("/generate_stream")
async def generate_stream(request: GenerationRequest):
    """
    Generate a streaming response to a user query.
    """
    if not request.stream:
        request.stream = True # Force stream for this endpoint

    logger.info(f"Received streaming query: '{request.query}' with provider: {request.provider}")

    try:
        response_generator = await llm_wrapper.generate(
            prompt=request.query,
            provider=request.provider,
            stream=True
        )

        if not isinstance(response_generator, AsyncGenerator):
            raise HTTPException(status_code=500, detail="Expected a streaming response but got a single response.")
        
        async def stream_generator():
            async for chunk in response_generator:
                yield chunk

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error during streaming generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    server_config = config.get('server', {})
    uvicorn.run(
        "app.main:app", 
        host=server_config.get('host', "0.0.0.0"), 
        port=server_config.get('port', 8000), 
        reload=True,
        reload_excludes=["logs/*"]
    ) 