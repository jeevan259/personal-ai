"""
Unit tests for Speech-to-Text
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock


class TestWhisperSTT:
    """Test Whisper STT implementation"""
    
    @pytest.fixture
    def stt(self):
        """Create STT instance for testing"""
        from speech.stt.whisper_stt import WhisperSTT
        return WhisperSTT(model_size="tiny", device="cpu")
        
    @pytest.mark.asyncio
    async def test_initialize(self, stt):
        """Test STT initialization"""
        with patch('speech.stt.whisper_stt.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            await stt.initialize()
            
            # Check model was loaded
            mock_load.assert_called_once_with("tiny", device="cpu")
            assert stt.model is mock_model
            
    @pytest.mark.asyncio
    async def test_transcribe_mock(self, stt):
        """Test transcription with mock"""
        # Mock whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "language": "en",
            "confidence": 0.9
        }
        stt.model = mock_model
        
        # Create test audio
        audio_data = np.random.randn(16000)  # 1 second at 16kHz
        
        # Transcribe
        text, metadata = await stt.transcribe(audio_data)
        
        # Check results
        assert text == "Hello world"
        assert metadata["language"] == "en"
        assert metadata["confidence"] == 0.9
        assert metadata["model"] == "tiny"
        
        # Verify model was called
        mock_model.transcribe.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self, stt):
        """Test transcription with empty audio"""
        audio_data = np.array([], dtype=np.float32)
        
        text, metadata = await stt.transcribe(audio_data)
        
        # Should handle empty audio gracefully
        assert text == ""
        assert "error" in metadata or metadata.get("confidence", 0) == 0
        
    def test_supported_languages(self, stt):
        """Test supported languages list"""
        languages = stt.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "en" in languages  # English should be supported
        
    @pytest.mark.asyncio
    async def test_transcribe_file(self, stt):
        """Test file transcription"""
        with patch('speech.stt.whisper_stt.whisper.load_model') as mock_load:
            mock_model = Mock()
            mock_model.transcribe.return_value = {
                "text": "Test transcription",
                "language": "en"
            }
            mock_load.return_value = mock_model
            
            stt.model = mock_model
            
            text, metadata = await stt.transcribe_file("test.wav")
            
            assert text == "Test transcription"
            assert metadata["language"] == "en"
            
    def test_cleanup(self, stt):
        """Test cleanup"""
        stt.model = Mock()
        stt.cleanup()
        
        assert stt.model is None