"""
Wake word detection using Porcupine (Picovoice)
"""

import numpy as np
from typing import List, Optional, Callable
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class PorcupineWakeWordDetector:
    """Wake word detection using Porcupine"""
    
    def __init__(
        self,
        access_key: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        sensitivities: Optional[List[float]] = None,
        model_path: Optional[Path] = None
    ):
        self.access_key = access_key
        self.keywords = keywords or ["hey assistant"]
        self.sensitivities = sensitivities or [0.5]
        self.model_path = model_path
        
        self.porcupine = None
        self.is_initialized = False
        
        # Callback for wake word detection
        self.detection_callback: Optional[Callable[[str], None]] = None
        
        logger.info(f"Wake word detector initialized for keywords: {keywords}")
        
    async def initialize(self):
        """Initialize Porcupine (async)"""
        if self.is_initialized:
            return
            
        try:
            # Try to import porcupine
            import pvporcupine
            
            # Create keyword paths
            keyword_paths = []
            for keyword in self.keywords:
                # This would need actual .ppn files
                # For now, use built-in if available
                try:
                    keyword_paths.append(
                        pvporcupine.KEYWORD_PATHS[keyword.replace(' ', '_').upper()]
                    )
                except:
                    logger.warning(f"Keyword '{keyword}' not found in built-in keywords")
                    # In real implementation, you'd provide path to .ppn file
                    
            if not keyword_paths:
                logger.error("No valid keyword paths found")
                return
                
            # Initialize Porcupine
            self.porcupine = pvporcupine.create(
                access_key=self.access_key or "mock_key_for_development",
                keywords=self.keywords,
                sensitivities=self.sensitivities,
                model_path=str(self.model_path) if self.model_path else None
            )
            
            self.is_initialized = True
            logger.info("Porcupine initialized successfully")
            
        except ImportError:
            logger.warning("pvporcupine not installed, using mock wake word detector")
            self.is_initialized = True  # Mark as initialized for mock
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            
    def process_audio(self, audio_frame: np.ndarray) -> Optional[str]:
        """
        Process audio frame for wake word detection
        
        Args:
            audio_frame: Audio frame (must be 16-bit PCM, mono)
            
        Returns:
            Detected keyword name or None
        """
        if not self.is_initialized:
            logger.warning("Wake word detector not initialized")
            return None
            
        if self.porcupine is None:
            # Mock implementation
            return self._mock_detection(audio_frame)
            
        try:
            # Convert to int16 if needed
            if audio_frame.dtype != np.int16:
                audio_frame = audio_frame.astype(np.int16)
                
            # Porcupine requires specific frame length
            if len(audio_frame) != self.porcupine.frame_length:
                logger.warning(f"Audio frame length mismatch: {len(audio_frame)} != {self.porcupine.frame_length}")
                return None
                
            # Process with Porcupine
            keyword_index = self.porcupine.process(audio_frame)
            
            if keyword_index >= 0:
                keyword = self.keywords[keyword_index]
                logger.info(f"Wake word detected: {keyword}")
                
                # Call callback if set
                if self.detection_callback:
                    try:
                        self.detection_callback(keyword)
                    except Exception as e:
                        logger.error(f"Detection callback error: {e}")
                        
                return keyword
                
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            
        return None
        
    def _mock_detection(self, audio_frame: np.ndarray) -> Optional[str]:
        """Mock wake word detection for testing"""
        # Simple energy-based detection
        energy = np.mean(np.abs(audio_frame))
        
        # Random detection for testing
        import random
        if energy > 5000 and random.random() < 0.01:  # 1% chance when loud
            keyword = self.keywords[0] if self.keywords else "hey assistant"
            logger.info(f"Mock wake word detected: {keyword}")
            
            if self.detection_callback:
                try:
                    self.detection_callback(keyword)
                except Exception as e:
                    logger.error(f"Detection callback error: {e}")
                    
            return keyword
            
        return None
        
    def set_detection_callback(self, callback: Callable[[str], None]):
        """Set callback for wake word detection"""
        self.detection_callback = callback
        
    async def train_custom_wake_word(
        self,
        samples_dir: Path,
        output_dir: Path,
        wake_word: str = "hey assistant"
    ) -> Optional[Path]:
        """
        Train custom wake word (placeholder implementation)
        
        Args:
            samples_dir: Directory with audio samples
            output_dir: Output directory for model
            wake_word: Wake word phrase
            
        Returns:
            Path to trained model or None
        """
        logger.info(f"Training custom wake word '{wake_word}' from {samples_dir}")
        
        # This would integrate with Picovoice's training service
        # For now, just create a placeholder
        
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{wake_word.replace(' ', '_')}.ppn"
        
        # Create dummy file
        with open(model_path, 'w') as f:
            f.write(f"Placeholder model for '{wake_word}'")
            
        logger.info(f"Mock model saved to: {model_path}")
        return model_path
        
    def get_required_frame_length(self) -> int:
        """Get required audio frame length"""
        if self.porcupine:
            return self.porcupine.frame_length
        return 512  # Default for mock
        
    def get_required_sample_rate(self) -> int:
        """Get required sample rate"""
        if self.porcupine:
            return self.porcupine.sample_rate
        return 16000  # Default for mock
        
    def cleanup(self):
        """Clean up resources"""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
            
        self.is_initialized = False
        logger.info("Wake word detector cleaned up")