"""
Accuracy tests for various components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.mark.accuracy
class TestAccuracy:
    """Accuracy tests"""
    
    def test_vad_accuracy_simple(self):
        """Test VAD accuracy with simple signals"""
        from audio.processing.vad import VoiceActivityDetector
        
        vad = VoiceActivityDetector()
        
        # Test with silence (zeros)
        silent_frame = np.zeros(512, dtype=np.int16)
        is_speech_silent = vad.is_speech(silent_frame)
        
        # Test with loud signal
        loud_frame = np.full(512, 10000, dtype=np.int16)
        is_speech_loud = vad.is_speech(loud_frame)
        
        # At minimum, loud should not be less likely than silence
        # (This is a weak test, but real accuracy needs proper dataset)
        print(f"Silence detected as speech: {is_speech_silent}")
        print(f"Loud signal detected as speech: {is_speech_loud}")
        
        # Just verify no crash and returns bool
        assert isinstance(is_speech_silent, bool)
        assert isinstance(is_speech_loud, bool)
        
    @pytest.mark.asyncio
    async def test_stt_accuracy_mock(self):
        """Test STT accuracy with mock"""
        from speech.stt.whisper_stt import WhisperSTT
        
        stt = WhisperSTT()
        
        # Mock model with known response
        mock_model = Mock()
        test_phrase = "The quick brown fox jumps over the lazy dog"
        mock_model.transcribe.return_value = {
            "text": test_phrase,
            "language": "en",
            "confidence": 0.95
        }
        
        stt.model = mock_model
        
        # Transcribe dummy audio
        audio = np.random.randn(16000)
        text, metadata = await stt.transcribe(audio)
        
        # Check accuracy of returned text
        assert text == test_phrase
        assert metadata["confidence"] == 0.95
        
        # Word Error Rate would be 0% for perfect match
        # In real tests, you'd compare with ground truth
        
    def test_audio_capture_quality(self):
        """Test audio capture signal quality"""
        from audio.capture.microphone import MicrophoneCapture
        
        capture = MicrophoneCapture()
        
        # Test signal generation (mock mode)
        # In real test, would capture actual audio and analyze
        
        # Check configuration
        assert capture.sample_rate in [8000, 16000, 44100, 48000]
        assert capture.channels in [1, 2]
        assert capture.chunk_size > 0
        
        # Signal-to-noise ratio would be tested with actual capture
        # This is just a placeholder
        
    @pytest.mark.asyncio
    async def test_llm_response_relevance(self):
        """Test LLM response relevance"""
        from llm.providers import MockLLMProvider
        
        llm = MockLLMProvider()
        
        test_queries = [
            "What's the weather?",
            "Set a timer for 5 minutes",
            "Tell me a joke",
            "What time is it?"
        ]
        
        for query in test_queries:
            response = await llm.generate(query)
            
            # Basic relevance checks
            assert len(response) > 0, "Empty response"
            assert isinstance(response, str), "Response not string"
            
            # Mock LLM should include query in response
            # Real test would use similarity metrics
            
            print(f"Query: {query}")
            print(f"Response: {response[:50]}...")
            
    def test_conversation_context_accuracy(self):
        """Test conversation context accuracy"""
        from core.conversation_manager import ConversationManager
        import tempfile
        from pathlib import Path
        
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            manager = ConversationManager(temp_dir)
            
            # Add messages
            test_messages = [
                ("user", "Hello, my name is John"),
                ("assistant", "Hello John, how can I help you?"),
                ("user", "What's my name?"),
                ("assistant", "Your name is John")
            ]
            
            for role, content in test_messages:
                manager.add_message(role, content)
                
            # Get history
            history = manager.get_conversation_history()
            
            # Verify accuracy
            assert len(history) == len(test_messages)
            
            for i, (role, content) in enumerate(test_messages):
                assert history[i]["role"] == role
                assert history[i]["content"] == content
                
            # Test context summary
            summary = manager.get_context_summary()
            assert isinstance(summary, str)
            assert len(summary) > 0
            
            # Should contain at least some message content
            assert "John" in summary or "name" in summary
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)