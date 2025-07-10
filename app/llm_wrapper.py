"""
LLM Wrapper for Ollama and Gemini integration with fallback logic
"""

import asyncio
import logging
import os
import time
from typing import AsyncGenerator, Dict, List, Optional, Union
from enum import Enum

import google.generativeai as genai
from google.generativeai.types import GenerationConfigDict
import ollama
from pydantic import BaseModel
import yaml
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    LOCAL = "local"
    GEMINI = "gemini"
    AUTO = "auto"


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    latency: float
    error: Optional[str] = None


class LLMWrapper:
    def __init__(self, config_path: str = "app/config.yaml"):
        """Initialize LLM wrapper with configuration"""
        self.config = self._load_config(config_path)
        self.gemini_client: Optional[genai.GenerativeModel] = None
        self._setup_gemini()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'llm': {
                'default_model': 'local',
                'local': {
                    'provider': 'ollama',
                    'model': 'gemma3:4b-it-qat',
                    'base_url': 'http://localhost:11434',
                    'timeout': 30
                },
                'gemini': {
                    'model': 'gemini-2.0-flash-exp',
                    'api_key_env': 'GEMINI_API_KEY',
                    'temperature': 0.7,
                    'max_tokens': 2048
                }
            }
        }
    
    def _setup_gemini(self):
        """Setup Gemini client"""
        try:
            api_key = os.getenv(self.config['llm']['gemini']['api_key_env'])
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_client = genai.GenerativeModel(
                    self.config['llm']['gemini']['model']
                )
                logger.info("Gemini client initialized successfully")
            else:
                logger.warning("Gemini API key not found")
        except Exception as e:
            logger.error(f"Failed to setup Gemini client: {e}")
    
    async def check_ollama_availability(self) -> bool:
        """Check if Ollama is available"""
        try:
            # Test connection to Ollama
            response = ollama.list()
            models = response.get('models', [])
            target_model = self.config['llm']['local']['model']
            
            # Check if the target model is available
            available_models = [model['name'] for model in models]
            if target_model in available_models:
                logger.info(f"Ollama model {target_model} is available")
                return True
            else:
                logger.warning(f"Model {target_model} not found. Available: {available_models}")
                return False
                
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    async def generate_ollama(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        start_time = time.time()
        
        try:
            model = self.config['llm']['local']['model']
            
            response = ollama.generate(
                model=model,
                prompt=prompt,
                stream=False,
                **kwargs
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=response['response'],
                provider="ollama",
                model=model,
                tokens_used=response.get('eval_count', 0) + response.get('prompt_eval_count', 0),
                latency=latency
            )
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Ollama generation failed: {e}")
            return LLMResponse(
                content="",
                provider="ollama",
                model=self.config['llm']['local']['model'],
                latency=latency,
                error=str(e)
            )
    
    async def generate_gemini(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Gemini"""
        start_time = time.time()
        
        try:
            assert self.gemini_client
            if not self.gemini_client:
                raise Exception("Gemini client not initialized")
            
            # Configure generation parameters
            generation_config = GenerationConfigDict(
                temperature=kwargs.get('temperature', self.config['llm']['gemini']['temperature']),
                max_output_tokens=kwargs.get('max_tokens', self.config['llm']['gemini']['max_tokens']),
            )
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini_client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=response.text,
                provider="gemini",
                model=self.config['llm']['gemini']['model'],
                tokens_used=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None,
                latency=latency
            )
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Gemini generation failed: {e}")
            return LLMResponse(
                content="",
                provider="gemini",
                model=self.config['llm']['gemini']['model'],
                latency=latency,
                error=str(e)
            )
    
    async def generate_streaming_ollama(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Ollama"""
        try:
            model = self.config['llm']['local']['model']
            
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                **kwargs
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"Error: {str(e)}"
    
    async def generate_streaming_gemini(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Generate streaming response using Gemini"""
        try:
            assert self.gemini_client
            if not self.gemini_client:
                raise Exception("Gemini client not initialized")

            generation_config = GenerationConfigDict(
                temperature=kwargs.get('temperature', self.config['llm']['gemini']['temperature']),
                max_output_tokens=kwargs.get('max_tokens', self.config['llm']['gemini']['max_tokens']),
            )

            # Use generate_content with stream=True for real streaming
            response_stream = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.gemini_client.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=True
                )
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield f"Error: {str(e)}"
    
    async def generate(
        self, 
        prompt: str, 
        provider: Optional[LLMProvider] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """
        Generate response with automatic fallback logic
        
        Args:
            prompt: Input prompt
            provider: Specific provider to use (local, gemini, auto)
            stream: Whether to return streaming response
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse or AsyncGenerator for streaming
        """
        if provider is None:
            provider = LLMProvider(self.config['llm']['default_model'])
        
        # Streaming responses
        if stream:
            return self._generate_stream(prompt, provider, **kwargs)
        
        # Non-streaming responses
        if provider == LLMProvider.LOCAL:
            if await self.check_ollama_availability():
                response = await self.generate_ollama(prompt, **kwargs)
                if response.error:
                    logger.warning("Local generation failed, trying Gemini fallback")
                    return await self.generate_gemini(prompt, **kwargs)
                return response
            else:
                logger.warning("Ollama not available, using Gemini")
                return await self.generate_gemini(prompt, **kwargs)
        
        elif provider == LLMProvider.GEMINI:
            return await self.generate_gemini(prompt, **kwargs)
        
        elif provider == LLMProvider.AUTO:
            # Try local first, fallback to Gemini
            if await self.check_ollama_availability():
                response = await self.generate_ollama(prompt, **kwargs)
                if not response.error:
                    return response
            
            logger.info("Falling back to Gemini")
            return await self.generate_gemini(prompt, **kwargs)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def _generate_stream(self, prompt: str, provider: LLMProvider, **kwargs) -> AsyncGenerator[str, None]:
        """Internal method to handle streaming generation"""
        if provider == LLMProvider.LOCAL:
            if await self.check_ollama_availability():
                async for chunk in self.generate_streaming_ollama(prompt, **kwargs):
                    yield chunk
            else:
                # Fallback to Gemini for streaming
                async for chunk in self.generate_streaming_gemini(prompt, **kwargs):
                    yield chunk
        elif provider == LLMProvider.GEMINI:
            async for chunk in self.generate_streaming_gemini(prompt, **kwargs):
                yield chunk
        elif provider == LLMProvider.AUTO:
            # Try local first, fallback to Gemini
            if await self.check_ollama_availability():
                async for chunk in self.generate_streaming_ollama(prompt, **kwargs):
                    yield chunk
            else:
                async for chunk in self.generate_streaming_gemini(prompt, **kwargs):
                    yield chunk


# Utility functions for easy access
async def generate_response(
    prompt: str, 
    provider: Optional[str] = None,
    stream: bool = False,
    **kwargs
) -> Union[LLMResponse, AsyncGenerator[str, None]]:
    """Convenience function for generating responses"""
    wrapper = LLMWrapper()
    provider_enum = LLMProvider(provider) if provider else None
    
    if stream:
        # Return the async generator directly
        return wrapper._generate_stream(prompt, provider_enum or LLMProvider.AUTO, **kwargs)
    else:
        # Return the LLMResponse
        return await wrapper.generate(prompt, provider_enum, stream, **kwargs)


if __name__ == "__main__":
    # Test the LLM wrapper
    async def test_llm():
        wrapper = LLMWrapper()
        
        test_prompt = "Explain what RAG (Retrieval Augmented Generation) is in simple terms."
        
        print("Testing Ollama...")
        try:
            response = await wrapper.generate(test_prompt, LLMProvider.LOCAL)
            if isinstance(response, LLMResponse):
                print(f"Response: {response.content[:100]}...")
                print(f"Provider: {response.provider}, Latency: {response.latency:.2f}s")
        except Exception as e:
            print(f"Ollama test failed: {e}")
        
        print("\nTesting Gemini...")
        try:
            response = await wrapper.generate(test_prompt, LLMProvider.GEMINI)
            if isinstance(response, LLMResponse):
                print(f"Response: {response.content[:100]}...")
                print(f"Provider: {response.provider}, Latency: {response.latency:.2f}s")
        except Exception as e:
            print(f"Gemini test failed: {e}")
        
        print("\nTesting Auto (with fallback)...")
        try:
            response = await wrapper.generate(test_prompt, LLMProvider.AUTO)
            if isinstance(response, LLMResponse):
                print(f"Response: {response.content[:100]}...")
                print(f"Provider: {response.provider}, Latency: {response.latency:.2f}s")
        except Exception as e:
            print(f"Auto test failed: {e}")
    
    asyncio.run(test_llm()) 