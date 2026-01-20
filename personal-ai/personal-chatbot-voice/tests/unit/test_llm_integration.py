"""
Unit tests for LLM integration
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestLLMProviders:
    """Test LLM providers"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock LLM config"""
        return {
            "providers": {
                "priority": ["local"],
                "local": {
                    "type": "ollama",
                    "model": "llama3.2",
                    "base_url": "http://localhost:11434"
                }
            }
        }
        
    def test_llm_provider_factory(self, mock_config):
        """Test LLM provider factory"""
        from llm.providers import LLMProviderFactory
        
        provider = LLMProviderFactory.create_provider(mock_config)
        
        # Should create a provider
        assert provider is not None
        assert hasattr(provider, 'generate')
        assert hasattr(provider, 'stream_chat')
        
    @pytest.mark.asyncio
    async def test_mock_llm_provider(self):
        """Test mock LLM provider"""
        from llm.providers import MockLLMProvider
        
        provider = MockLLMProvider()
        
        # Test generate
        response = await provider.generate("Hello")
        assert "mock" in response.lower()
        
        # Test stream
        chunks = []
        async for chunk in provider.stream_chat([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)
            
        assert len(chunks) > 0
        assert "mock" in "".join(chunks).lower()
        
        # Test model info
        info = provider.get_model_info()
        assert info["provider"] == "mock"
        
    @pytest.mark.asyncio
    async def test_local_ollama_provider(self):
        """Test Ollama provider (mocked)"""
        from llm.providers import LocalOllamaProvider
        
        provider = LocalOllamaProvider()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "response": "Hello from Ollama"
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            response = await provider.generate("Hello")
            
            assert response == "Hello from Ollama"
            
    def test_provider_info(self):
        """Test provider information"""
        from llm.providers import LocalOllamaProvider, MockLLMProvider
        
        ollama_info = LocalOllamaProvider().get_model_info()
        assert ollama_info["provider"] == "ollama"
        assert ollama_info["type"] == "local"
        
        mock_info = MockLLMProvider().get_model_info()
        assert mock_info["provider"] == "mock"