"""
Whisper STT implementation for speech-to-text
"""

import numpy as np
import whisper
from typing import Optional, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)

class WhisperSTT:
    """Speech-to-Text using OpenAI Whisper"""
    
    def __init__(self, model_size: str = "base", device: str = "cpu", language: str = "en"):
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        
    async def initialize(self):
        """Initialize Whisper model (async)"""
        if self.model is None:
            try:
                # Run model loading in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, self._load_model
                )
                logger.info(f"Whisper model '{self.model_size}' loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                raise
                
    def _load_model(self):
        """Load Whisper model (synchronous)"""
        return whisper.load_model(self.model_size, device=self.device)
        
    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Tuple[str, dict]:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (transcribed_text, metadata)
        """
        if self.model is None:
            await self.initialize()
            
        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_sync, audio_data
            )
            
            text = result.get("text", "").strip()
            language = result.get("language", "en")
            confidence = result.get("confidence", 0.0)
            
            logger.debug(f"Transcribed: {text[:50]}... (lang: {language}, confidence: {confidence})")
            
            return text, {
                "language": language,
                "confidence": confidence,
                "model": self.model_size
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", {"error": str(e)}
            
    def _transcribe_sync(self, audio_data: np.ndarray) -> dict:
        """Synchronous transcription"""
        return self.model.transcribe(
            audio_data,
            language=self.language,
            fp16=False  # Disable FP16 for CPU compatibility
        )
        
    async def transcribe_file(self, file_path: str) -> Tuple[str, dict]:
        """Transcribe audio file"""
        if self.model is None:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self.model.transcribe, file_path, {"language": self.language}
            )
            
            text = result.get("text", "").strip()
            return text, {
                "language": result.get("language", "en"),
                "confidence": result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return "", {"error": str(e)}
            
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        # Whisper supports multilingual transcription
        return ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko", "ar", "hi"]
        
    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            # Whisper models don't have explicit cleanup
            self.model = None