"""
Unit tests for audio processing
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio

# Import modules to test
from audio.processing.vad import VoiceActivityDetector
from audio.capture.microphone import MicrophoneCapture


class TestVoiceActivityDetector:
    """Test Voice Activity Detection"""
    
    def setup_method(self):
        """Setup before each test"""
        self.vad = VoiceActivityDetector(sample_rate=16000)
        
    def test_init(self):
        """Test VAD initialization"""
        assert self.vad.sample_rate == 16000
        assert self.vad.aggressiveness == 3
        assert self.vad.frame_size > 0
        
    def test_mock_detection(self):
        """Test VAD with mock implementation"""
        # Create silent audio
        silent_audio = np.zeros(512, dtype=np.int16)
        
        # Create speech-like audio (sine wave)
        t = np.linspace(0, 0.1, 512)
        speech_audio = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        
        # Test with mock VAD (webrtcvad not installed)
        with patch('audio.processing.vad.webrtcvad', None):
            vad = VoiceActivityDetector()
            
            # Mock detection should work
            result1 = vad.is_speech(silent_audio)
            result2 = vad.is_speech(speech_audio)
            
            # At least test that it doesn't crash
            assert isinstance(result1, bool)
            assert isinstance(result2, bool)
            
    def test_process_audio_stream(self):
        """Test audio stream processing"""
        audio_frame = np.random.randint(-1000, 1000, 512, dtype=np.int16)
        
        segment, started, ended = self.vad.process_audio_stream(audio_frame)
        
        # Should return valid types
        assert segment is None or isinstance(segment, np.ndarray)
        assert isinstance(started, bool)
        assert isinstance(ended, bool)
        
    def test_reset(self):
        """Test VAD reset"""
        self.vad.speech_buffer = np.array([1, 2, 3])
        self.vad.silence_frames = 5
        
        self.vad.reset()
        
        assert self.vad.speech_buffer is None
        assert self.vad.silence_frames == 0


class TestMicrophoneCapture:
    """Test microphone capture"""
    
    def setup_method(self):
        """Setup before each test"""
        self.capture = MicrophoneCapture(sample_rate=16000, chunk_size=512)
        
    def test_init(self):
        """Test initialization"""
        assert self.capture.sample_rate == 16000
        assert self.capture.channels == 1
        assert self.capture.chunk_size == 512
        assert not self.capture.is_capturing
        
    @patch('audio.capture.microphone.pyaudio')
    def test_start_stop(self, mock_pyaudio):
        """Test start and stop capture"""
        # Mock PyAudio objects
        mock_stream = Mock()
        mock_audio = Mock()
        mock_audio.open.return_value = mock_stream
        mock_pyaudio.PyAudio.return_value = mock_audio
        
        # Start capture
        self.capture.start()
        
        # Check state
        assert self.capture.is_capturing
        assert self.capture.capture_thread is not None
        
        # Stop capture
        self.capture.stop()
        
        assert not self.capture.is_capturing
        
    def test_get_available_devices_mock(self):
        """Test getting available devices (mock mode)"""
        devices = self.capture.get_available_devices()
        
        # Should return list
        assert isinstance(devices, list)
        
        # Mock should return some devices
        assert len(devices) > 0
        for device in devices:
            assert 'index' in device
            assert 'name' in device
            assert 'channels' in device
            assert 'sample_rate' in device
            
    @pytest.mark.asyncio
    async def test_get_audio_chunk(self):
        """Test async audio chunk retrieval"""
        # Mock queue
        test_audio = np.array([1, 2, 3], dtype=np.int16)
        self.capture.audio_queue.put(test_audio)
        
        # Get chunk
        chunk = await self.capture.get_audio_chunk(timeout=0.1)
        
        assert chunk is not None
        assert np.array_equal(chunk, test_audio)


@pytest.mark.asyncio
class TestAudioAsync:
    """Async audio tests"""
    
    async def test_async_audio_flow(self):
        """Test async audio flow"""
        capture = MicrophoneCapture()
        vad = VoiceActivityDetector()
        
        # Create test audio
        test_audio = np.random.randint(-1000, 1000, 512, dtype=np.int16)
        
        # Process through VAD
        segment, started, ended = vad.process_audio_stream(test_audio)
        
        # Just check no exceptions
        assert True