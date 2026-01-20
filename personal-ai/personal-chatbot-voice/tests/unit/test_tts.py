"""
Unit tests for Text-to-Speech
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestPyTTSX3TTS:
    """Test PyTTSX3 TTS implementation"""
    
    @pytest.fixture
    def tts(self):
        """Create TTS instance for testing"""
        from speech.tts.pyttsx3_local import PyTTSX3TTS
        return PyTTSX3TTS(rate=150, volume=0.8)
        
    def test_init(self, tts):
        """Test TTS initialization"""
        assert tts.engine is not None
        # Can't easily test pyttsx3 internals, just verify no crash
        
    @pytest.mark.asyncio
    async def test_speak(self, tts):
        """Test speak method"""
        with patch.object(tts.engine, 'say') as mock_say, \
             patch.object(tts.engine, 'runAndWait') as mock_run:
            
            await tts.speak("Hello world")
            
            # Verify methods were called
            mock_say.assert_called_once_with("Hello world")
            mock_run.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_speak_async(self, tts):
        """Test async speak without waiting"""
        with patch.object(tts.engine, 'say') as mock_say, \
             patch.object(tts.engine, 'runAndWait') as mock_run:
            
            # Don't wait for completion
            await tts.speak("Test", wait=False)
            
            # Should still call say
            mock_say.assert_called_once_with("Test")
            # runAndWait might be called in thread
            
    def test_get_available_voices(self, tts):
        """Test getting available voices"""
        # Mock the engine's getProperty method
        mock_voices = [
            Mock(id="voice1", name="English Male", languages=["en"], gender="male"),
            Mock(id="voice2", name="English Female", languages=["en"], gender="female"),
        ]
        
        with patch.object(tts.engine, 'getProperty') as mock_get:
            mock_get.return_value = mock_voices
            
            voices = tts.get_available_voices()
            
            assert len(voices) == 2
            assert voices[0]["name"] == "English Male"
            assert voices[0]["gender"] == "male"
            assert voices[0]["languages"] == ["en"]
            
    def test_change_voice(self, tts):
        """Test changing voice"""
        mock_voices = [
            Mock(id="voice1", name="Voice 1"),
            Mock(id="voice2", name="Voice 2"),
        ]
        
        with patch.object(tts.engine, 'getProperty') as mock_get, \
             patch.object(tts.engine, 'setProperty') as mock_set:
            
            mock_get.return_value = mock_voices
            
            # Change to voice 1
            result = tts.change_voice(0)
            assert result is True
            mock_set.assert_called_with('voice', 'voice1')
            
            # Change to invalid voice
            result = tts.change_voice(10)
            assert result is False
            
    def test_stop(self, tts):
        """Test stop method"""
        with patch.object(tts.engine, 'stop') as mock_stop:
            tts.stop()
            mock_stop.assert_called_once()
            
    def test_cleanup(self, tts):
        """Test cleanup"""
        # Should not crash
        tts.cleanup()