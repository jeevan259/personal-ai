"""
Wake word detection manager
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np

from src.audio.capture.microphone import MicrophoneCapture
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class WakeWordManager:
    """Manages wake word detection using multiple engines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Wake word engine
        self.engine_type = config.get('engine', 'porcupine')
        self.wake_engine = None
        
        # Detection state
        self.is_detecting = False
        self.last_detection_time = 0
        self.cooldown_seconds = config.get('cooldown', 2.0)
        
        # Callbacks
        self.on_wake_detected = None
        self.on_error = None
        
        # Audio capture
        self.microphone = None
        
    async def initialize(self):
        """Initialize wake word detection"""
        logger.info(f"Initializing wake word manager with engine: {self.engine_type}")
        
        try:
            # Initialize the appropriate wake word engine
            if self.engine_type == 'porcupine':
                self.wake_engine = await self._init_porcupine()
            elif self.engine_type == 'custom':
                self.wake_engine = await self._init_custom_wake_word()
            else:
                raise ValueError(f"Unknown wake word engine: {self.engine_type}")
            
            # Initialize microphone
            self.microphone = MicrophoneCapture(self.config.get('audio', {}))
            await self.microphone.initialize()
            
            logger.info("Wake word manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize wake word manager: {e}", exc_info=True)
            return False
    
    async def _init_porcupine(self):
        """Initialize Porcupine wake word engine"""
        try:
            import pvporcupine
            
            access_key = self.config.get('porcupine', {}).get('access_key')
            if not access_key:
                logger.warning("No Porcupine access key provided")
                return None
            
            # Get keywords and sensitivities
            keywords = self.config.get('porcupine', {}).get('keywords', ['computer'])
            sensitivities = self.config.get('porcupine', {}).get('sensitivities', [0.7])
            
            # Create Porcupine instance
            porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=keywords,
                sensitivities=sensitivities
            )
            
            logger.info(f"Porcupine initialized with keywords: {keywords}")
            return porcupine
            
        except ImportError:
            logger.error("Porcupine not installed. Install with: pip install pvporcupine")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            return None
    
    async def _init_custom_wake_word(self):
        """Initialize custom wake word engine"""
        try:
            model_path = self.config.get('custom', {}).get('model_path')
            if not model_path:
                logger.error("No custom wake word model path provided")
                return None
            
            # This would load a custom trained model
            # For now, return a placeholder
            logger.info(f"Custom wake word model would be loaded from: {model_path}")
            return {
                'type': 'custom',
                'model_path': model_path,
                'threshold': self.config.get('custom', {}).get('threshold', 0.7)
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize custom wake word: {e}")
            return None
    
    async def start_detection(self, on_wake_detected: Callable, on_error: Optional[Callable] = None):
        """Start wake word detection"""
        if self.is_detecting:
            logger.warning("Wake word detection already running")
            return
        
        if not self.wake_engine:
            logger.error("Wake word engine not initialized")
            return
        
        self.on_wake_detected = on_wake_detected
        self.on_error = on_error
        self.is_detecting = True
        
        # Start detection in background task
        asyncio.create_task(self._detection_loop())
        
        logger.info("Wake word detection started")
    
    async def _detection_loop(self):
        """Main detection loop"""
        if not self.microphone:
            logger.error("Microphone not initialized")
            return
        
        # Start microphone
        self.microphone.start_recording()
        
        try:
            while self.is_detecting:
                # Read audio chunk
                audio_chunk = self.microphone.read_chunk(timeout=0.1)
                
                if audio_chunk is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process for wake word
                detected = await self._process_audio_chunk(audio_chunk)
                
                if detected:
                    await self._handle_wake_detection(detected)
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in detection loop: {e}", exc_info=True)
            if self.on_error:
                self.on_error(e)
        finally:
            self.microphone.stop_recording()
    
    async def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process audio chunk for wake word detection"""
        
        if self.engine_type == 'porcupine' and self.wake_engine:
            return await self._process_with_porcupine(audio_chunk)
        elif self.engine_type == 'custom' and self.wake_engine:
            return await self._process_custom_wake_word(audio_chunk)
        
        return None
    
    async def _process_with_porcupine(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process audio with Porcupine"""
        try:
            import pvporcupine
            
            # Ensure audio is in correct format
            if audio_chunk.dtype != np.int16:
                audio_chunk = audio_chunk.astype(np.int16)
            
            # Check if audio chunk is the right size
            frame_length = self.wake_engine.frame_length
            if len(audio_chunk) != frame_length:
                # Resize or pad
                if len(audio_chunk) > frame_length:
                    audio_chunk = audio_chunk[:frame_length]
                else:
                    padding = np.zeros(frame_length - len(audio_chunk), dtype=np.int16)
                    audio_chunk = np.concatenate([audio_chunk, padding])
            
            # Process with Porcupine
            keyword_index = self.wake_engine.process(audio_chunk)
            
            if keyword_index >= 0:
                keyword = self.wake_engine.keywords[keyword_index]
                return {
                    'keyword': keyword,
                    'confidence': 1.0,  # Porcupine doesn't provide confidence
                    'timestamp': asyncio.get_event_loop().time(),
                    'engine': 'porcupine'
                }
        
        except Exception as e:
            logger.error(f"Error processing with Porcupine: {e}")
        
        return None
    
    async def _process_custom_wake_word(self, audio_chunk: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process audio with custom wake word detector"""
        # Placeholder for custom wake word processing
        # In production, this would load and run a custom model
        
        return None
    
    async def _handle_wake_detection(self, detection: Dict[str, Any]):
        """Handle wake word detection"""
        import time
        
        # Check cooldown
        current_time = time.time()
        time_since_last = current_time - self.last_detection_time
        
        if time_since_last < self.cooldown_seconds:
            logger.debug(f"Ignoring wake word (cooldown: {time_since_last:.2f}s)")
            return
        
        # Update last detection time
        self.last_detection_time = current_time
        
        logger.info(f"Wake word detected: {detection.get('keyword', 'Unknown')}")
        
        # Play confirmation sound if configured
        if self.config.get('confirmation_beep', True):
            await self._play_confirmation_beep()
        
        # Call callback
        if self.on_wake_detected:
            await self.on_wake_detected(detection)
    
    async def _play_confirmation_beep(self):
        """Play a confirmation beep sound"""
        try:
            import sounddevice as sd
            
            # Generate a simple beep
            sample_rate = 44100
            duration = 0.1
            frequency = 1000
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Play the beep
            sd.play(beep, sample_rate)
            sd.wait()
            
        except ImportError:
            logger.warning("sounddevice not installed, skipping beep")
        except Exception as e:
            logger.error(f"Failed to play confirmation beep: {e}")
    
    async def stop_detection(self):
        """Stop wake word detection"""
        if not self.is_detecting:
            return
        
        self.is_detecting = False
        
        if self.microphone:
            self.microphone.stop_recording()
        
        logger.info("Wake word detection stopped")
    
    async def add_custom_wake_word(self, wake_word: str, audio_samples: list):
        """Add a custom wake word"""
        # This would train and save a custom wake word model
        # Placeholder implementation
        
        logger.info(f"Adding custom wake word: {wake_word}")
        
        # Save training data
        training_dir = Path("data/audio_samples/wake_word") / wake_word
        training_dir.mkdir(parents=True, exist_ok=True)
        
        for i, sample in enumerate(audio_samples):
            sample_path = training_dir / f"sample_{i}.wav"
            # Save audio sample
            # In production, use a proper audio library
        
        # Train model
        # This would call a training script
        
        logger.info(f"Custom wake word '{wake_word}' added")
    
    async def get_detection_stats(self) -> Dict[str, Any]:
        """Get wake word detection statistics"""
        return {
            'engine': self.engine_type,
            'is_detecting': self.is_detecting,
            'detections_today': 0,  # Would be tracked
            'false_positives': 0,
            'accuracy': 1.0
        }
    
    async def stop(self):
        """Cleanup resources"""
        await self.stop_detection()
        
        if self.wake_engine and hasattr(self.wake_engine, 'delete'):
            self.wake_engine.delete()
            self.wake_engine = None
        
        if self.microphone:
            await self.microphone.stop()
            self.microphone = None