"""
PyTTSX3 Text-to-Speech implementation
Simple, offline TTS for development
"""

import pyttsx3
import asyncio
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PyTTSX3TTS:
    """Text-to-Speech using PyTTSX3 (offline, no API needed)"""
    
    def __init__(self, rate: int = 150, volume: float = 0.9, voice_id: Optional[int] = None):
        self.engine = pyttsx3.init()
        
        # Configure voice settings
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Select voice
        voices = self.engine.getProperty('voices')
        if voice_id is not None and voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)
        else:
            # Use default voice
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
                    
        logger.info(f"TTS initialized with rate={rate}, volume={volume}")
        
    async def speak(self, text: str, wait: bool = True) -> None:
        """Convert text to speech"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            if wait:
                await loop.run_in_executor(None, self._speak_sync, text)
            else:
                # Start speaking without waiting
                loop.run_in_executor(None, self._speak_sync, text)
                
            logger.debug(f"Spoke: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            raise
            
    def _speak_sync(self, text: str) -> None:
        """Synchronous speak method for thread pool"""
        self.engine.say(text)
        self.engine.runAndWait()
        
    def stop(self) -> None:
        """Stop current speech"""
        self.engine.stop()
        
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        voices = self.engine.getProperty('voices')
        return [
            {
                "id": i,
                "name": voice.name,
                "languages": voice.languages,
                "gender": voice.gender,
            }
            for i, voice in enumerate(voices)
        ]
        
    def change_voice(self, voice_id: int) -> bool:
        """Change to a different voice"""
        voices = self.engine.getProperty('voices')
        if 0 <= voice_id < len(voices):
            self.engine.setProperty('voice', voices[voice_id].id)
            logger.info(f"Changed voice to: {voices[voice_id].name}")
            return True
        return False
        
    def cleanup(self):
        """Clean up resources"""
        self.engine.stop()
        # PyTTSX3 doesn't have explicit cleanup
        
    def __del__(self):
        """Destructor"""
        self.cleanup()