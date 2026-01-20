"""
LLM Provider factory and base implementations
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncGenerator
import logging
import asyncio

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        pass
        
    @abstractmethod
    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response"""
        pass
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model"""
        pass


class LocalOllamaProvider(LLMProvider):
    """Local LLM using Ollama"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        return f"Error: {response.status}"
                        
        except ImportError:
            logger.error("aiohttp not installed")
            return "Error: aiohttp not installed"
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"Error: {str(e)}"
            
    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response from Ollama"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": True,
                    **kwargs
                }
                
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                ) as response:
                    if response.status == 200:
                        async for line in response.content:
                            if line:
                                try:
                                    decoded = line.decode('utf-8').strip()
                                    if decoded:
                                        # Parse SSE format
                                        if decoded.startswith('data: '):
                                            data = decoded[6:]
                                            if data != '[DONE]':
                                                import json
                                                chunk = json.loads(data)
                                                if 'message' in chunk and 'content' in chunk['message']:
                                                    yield chunk['message']['content']
                                except Exception as e:
                                    logger.debug(f"Error parsing stream chunk: {e}")
                                    continue
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        yield f"Error: {response.status}"
                        
        except ImportError:
            logger.error("aiohttp not installed")
            yield "Error: aiohttp not installed"
        except Exception as e:
            logger.error(f"Ollama streaming error: {e}")
            yield f"Error: {str(e)}"
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information"""
        return {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
            "type": "local"
        }


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""
    
    def __init__(self):
        self.model = "mock-llm"
        
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation"""
        logger.debug(f"Mock LLM received prompt: {prompt[:50]}...")
        return f"I'm a mock LLM. You said: {prompt}"
        
    async def stream_chat(self, messages: list, **kwargs) -> AsyncGenerator[str, None]:
        """Mock streaming"""
        response = "This is a mock response from the LLM. "
        response += "In a real implementation, this would stream from an actual model."
        
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.1)
            
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model info"""
        return {
            "provider": "mock",
            "model": "mock-llm",
            "type": "test"
        }


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(config: Dict[str, Any]) -> LLMProvider:
        """Create LLM provider based on configuration"""
        try:
            provider_type = config.get("providers", {}).get("priority", ["local"])[0]
            
            if provider_type == "local":
                local_config = config.get("providers", {}).get("local", {})
                if local_config.get("type") == "ollama":
                    return LocalOllamaProvider(
                        base_url=local_config.get("base_url", "http://localhost:11434"),
                        model=local_config.get("model", "llama3.2")
                    )
                    
            # Fallback to mock provider
            logger.warning(f"LLM provider {provider_type} not fully implemented, using mock")
            return MockLLMProvider()
            
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")
            return MockLLMProvider()