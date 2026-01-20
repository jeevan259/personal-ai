"""
Performance tests for latency
"""

import pytest
import time
import asyncio
import numpy as np
from unittest.mock import Mock, patch


@pytest.mark.performance
class TestLatencyPerformance:
    """Performance tests for latency-critical operations"""
    
    @pytest.fixture
    def performance_setup(self):
        """Setup for performance testing"""
        class TimingResult:
            def __init__(self):
                self.times = []
                
            def add_time(self, duration):
                self.times.append(duration)
                
            def get_stats(self):
                if not self.times:
                    return {}
                return {
                    "count": len(self.times),
                    "min": min(self.times),
                    "max": max(self.times),
                    "mean": sum(self.times) / len(self.times),
                    "total": sum(self.times)
                }
                
        return TimingResult()
        
    @pytest.mark.asyncio
    async def test_stt_latency(self, performance_setup):
        """Test STT latency"""
        from speech.stt.whisper_stt import WhisperSTT
        
        stt = WhisperSTT(model_size="tiny")
        
        # Mock the model for performance testing
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "performance test",
            "language": "en"
        }
        
        with patch('speech.stt.whisper_stt.whisper.load_model', return_value=mock_model):
            await stt.initialize()
            
            # Create test audio (1 second at 16kHz)
            audio_data = np.random.randn(16000).astype(np.float32)
            
            # Run multiple iterations
            iterations = 5
            for i in range(iterations):
                start_time = time.time()
                await stt.transcribe(audio_data)
                end_time = time.time()
                
                duration = end_time - start_time
                performance_setup.add_time(duration)
                
                # Each iteration should be reasonably fast
                assert duration < 2.0, f"STT too slow: {duration:.2f}s"
                
        stats = performance_setup.get_stats()
        print(f"STT Performance: {stats}")
        
        # Average should be under threshold
        assert stats["mean"] < 1.0, f"Average STT too slow: {stats['mean']:.2f}s"
        
    @pytest.mark.asyncio
    async def test_llm_response_latency(self, performance_setup):
        """Test LLM response latency"""
        from llm.providers import MockLLMProvider
        
        llm = MockLLMProvider()
        
        # Test multiple requests
        iterations = 10
        for i in range(iterations):
            start_time = time.time()
            await llm.generate(f"Test query {i}")
            end_time = time.time()
            
            duration = end_time - start_time
            performance_setup.add_time(duration)
            
            # Mock LLM should be very fast
            assert duration < 0.1, f"LLM too slow: {duration:.2f}s"
            
        stats = performance_setup.get_stats()
        print(f"LLM Performance: {stats}")
        
        assert stats["mean"] < 0.05, f"Average LLM too slow: {stats['mean']:.2f}s"
        
    @pytest.mark.asyncio
    async def test_full_pipeline_latency(self, performance_setup):
        """Test full pipeline latency"""
        # Create minimal mock pipeline
        class MockPipeline:
            async def process(self, audio):
                # Simulate pipeline processing
                await asyncio.sleep(0.05)  # STT
                await asyncio.sleep(0.1)   # LLM
                await asyncio.sleep(0.02)  # TTS prep
                return "response"
                
        pipeline = MockPipeline()
        
        # Test pipeline latency
        iterations = 5
        for i in range(iterations):
            start_time = time.time()
            await pipeline.process(None)
            end_time = time.time()
            
            duration = end_time - start_time
            performance_setup.add_time(duration)
            
            # Full pipeline should complete in reasonable time
            assert duration < 0.5, f"Pipeline too slow: {duration:.2f}s"
            
        stats = performance_setup.get_stats()
        print(f"Pipeline Performance: {stats}")
        
        # With mocks, should be fast
        assert stats["mean"] < 0.2, f"Average pipeline too slow: {stats['mean']:.2f}s"
        
    def test_audio_processing_latency(self, performance_setup):
        """Test audio processing latency"""
        from audio.processing.vad import VoiceActivityDetector
        
        vad = VoiceActivityDetector()
        
        # Generate test frames
        frame_size = vad.get_frame_size()
        iterations = 1000
        
        for i in range(iterations):
            audio_frame = np.random.randint(-1000, 1000, frame_size, dtype=np.int16)
            
            start_time = time.perf_counter()
            vad.is_speech(audio_frame)
            end_time = time.perf_counter()
            
            duration = (end_time - start_time) * 1000  # Convert to ms
            performance_setup.add_time(duration)
            
            # VAD should be very fast (sub-millisecond)
            assert duration < 10, f"VAD too slow: {duration:.2f}ms"
            
        stats = performance_setup.get_stats()
        print(f"VAD Performance: {stats}")
        
        # Average should be under 1ms
        assert stats["mean"] < 1.0, f"Average VAD too slow: {stats['mean']:.2f}ms"