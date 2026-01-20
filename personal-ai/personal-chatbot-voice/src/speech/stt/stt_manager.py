"""
Speech-to-Text Manager supporting multiple engines
"""

import sys
import os

# Fix imports - add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Now this import should work
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class STTManager:
    """Manages multiple STT engines with fallback support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Available STT engines
        self.engines = {}
        self.active_engine = None
        self.engine_order = config.get('engine_order', ['whisper', 'vosk', 'google'])
        
        # Performance tracking
        self.stats = {
            'total_transcriptions': 0,
            'successful_transcriptions': 0,
            'average_latency': 0.0
        }
        
        # Model paths
        self.model_paths = {
            'whisper': Path(config.get('whisper_model_path', 'models/whisper')),
            'vosk': Path(config.get('vosk_model_path', 'models/vosk'))
        }
        
    async def initialize(self):
        """Initialize STT engines"""
        logger.info("Initializing STT Manager...")
        
        try:
            # Initialize engines in order of preference
            for engine_name in self.engine_order:
                if await self._initialize_engine(engine_name):
                    logger.info(f"Initialized {engine_name} STT engine")
                    if self.active_engine is None:
                        self.active_engine = engine_name
                else:
                    logger.warning(f"Failed to initialize {engine_name} STT engine")
            
            if self.active_engine is None:
                raise RuntimeError("No STT engines could be initialized")
            
            logger.info(f"STT Manager initialized with active engine: {self.active_engine}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize STT Manager: {e}", exc_info=True)
            return False
    
    async def _initialize_engine(self, engine_name: str) -> bool:
        """Initialize a specific STT engine"""
        try:
            if engine_name == 'whisper':
                return await self._init_whisper()
            elif engine_name == 'vosk':
                return await self._init_vosk()
            elif engine_name == 'google':
                return await self._init_google()
            else:
                logger.warning(f"Unknown STT engine: {engine_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing {engine_name}: {e}")
            return False
    
    async def _init_whisper(self) -> bool:
        """Initialize Whisper STT"""
        try:
            import whisper
            
            model_size = self.config.get('whisper_model_size', 'base')
            logger.info(f"Loading Whisper {model_size} model...")
            
            # Load model (will download if not exists)
            model = whisper.load_model(model_size)
            
            self.engines['whisper'] = {
                'engine': model,
                'type': 'whisper',
                'supported_languages': ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh'],
                'realtime': False
            }
            
            return True
            
        except ImportError:
            logger.warning("Whisper not installed. Install with: pip install openai-whisper")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            return False
    
    async def _init_vosk(self) -> bool:
        """Initialize Vosk STT (offline)"""
        try:
            # Vosk is already installed (version 0.3.41)
            from vosk import Model, KaldiRecognizer
            import json
            
            model_path = self.model_paths['vosk']
            
            # Check if model exists, if not provide helpful message
            if not model_path.exists():
                logger.warning(f"Vosk model not found at {model_path}")
                logger.info("Download a vosk model from: https://alphacephei.com/vosk/models")
                logger.info("Example small English model: vosk-model-small-en-us-0.15")
                logger.info("Extract to: models/vosk/")
                return False
            
            logger.info(f"Loading Vosk model from {model_path}...")
            model = Model(str(model_path))
            
            self.engines['vosk'] = {
                'engine': model,
                'recognizer_class': KaldiRecognizer,
                'type': 'vosk',
                'supported_languages': ['en'],
                'realtime': True,
                'sample_rate': 16000
            }
            
            return True
            
        except ImportError:
            logger.warning("Vosk import failed. Make sure vosk-0.3.41 is properly installed.")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            return False
    
    async def _init_google(self) -> bool:
        """Initialize Google Speech-to-Text"""
        # Placeholder - Google STT requires API key
        logger.info("Google STT requires API key setup")
        return False
    
    async def transcribe(
        self,
        audio_data: np.ndarray,
        language: str = "en",
        engine: Optional[str] = None
    ) -> Optional[str]:
        """Transcribe audio to text"""
        
        if engine is None:
            engine = self.active_engine
        
        if engine not in self.engines:
            logger.error(f"STT engine not available: {engine}")
            # Try fallback
            for fallback_engine in self.engine_order:
                if fallback_engine in self.engines and fallback_engine != engine:
                    logger.info(f"Falling back to {fallback_engine}")
                    return await self.transcribe(audio_data, language, fallback_engine)
            return None
        
        engine_info = self.engines[engine]
        
        # Check language support
        if language not in engine_info.get('supported_languages', [language]):
            logger.warning(f"Language {language} may not be fully supported by {engine}")
        
        try:
            import time
            start_time = time.time()
            
            # Perform transcription
            if engine == 'whisper':
                transcript = await self._transcribe_whisper(audio_data, language)
            elif engine == 'vosk':
                transcript = await self._transcribe_vosk(audio_data, language)
            elif engine == 'google':
                transcript = await self._transcribe_google(audio_data, language)
            else:
                logger.error(f"Unsupported engine: {engine}")
                return None
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Update statistics
            self.stats['total_transcriptions'] += 1
            if transcript:
                self.stats['successful_transcriptions'] += 1
                
                # Update average latency (exponential moving average)
                alpha = 0.1
                self.stats['average_latency'] = (
                    alpha * latency + 
                    (1 - alpha) * self.stats['average_latency']
                )
            
            logger.debug(f"STT ({engine}) latency: {latency:.3f}s")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed with {engine}: {e}", exc_info=True)
            
            # Try fallback engine
            if engine != self.engine_order[0]:
                for fallback_engine in self.engine_order:
                    if (fallback_engine in self.engines and 
                        fallback_engine != engine and 
                        fallback_engine != self.active_engine):
                        logger.info(f"Trying fallback engine: {fallback_engine}")
                        return await self.transcribe(audio_data, language, fallback_engine)
            
            return None
    
    async def _transcribe_whisper(self, audio_data: np.ndarray, language: str) -> Optional[str]:
        """Transcribe using Whisper"""
        try:
            import whisper
            
            engine_info = self.engines['whisper']
            model = engine_info['engine']
            
            # Convert audio to float32
            if audio_data.dtype != np.float32:
                audio_float = audio_data.astype(np.float32) / 32768.0
            else:
                audio_float = audio_data
            
            # Transcribe
            result = model.transcribe(
                audio_float,
                language=language,
                fp16=False,  # Use FP32 for compatibility
                task='transcribe'
            )
            
            return result.get('text', '').strip()
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return None
    
    async def _transcribe_vosk(self, audio_data: np.ndarray, language: str) -> Optional[str]:
        """Transcribe using Vosk"""
        try:
            engine_info = self.engines['vosk']
            model = engine_info['engine']
            Recognizer = engine_info['recognizer_class']
            
            # Create recognizer
            recognizer = Recognizer(model, engine_info['sample_rate'])
            recognizer.SetWords(True)
            
            # Ensure audio is in correct format
            if audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            
            # Process in chunks
            chunk_size = 4000
            transcript_parts = []
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size].tobytes()
                
                if recognizer.AcceptWaveform(chunk):
                    result = recognizer.Result()
                    import json
                    result_dict = json.loads(result)
                    if 'text' in result_dict and result_dict['text']:
                        transcript_parts.append(result_dict['text'])
            
            # Get final result
            result = recognizer.FinalResult()
            import json
            result_dict = json.loads(result)
            if 'text' in result_dict and result_dict['text']:
                transcript_parts.append(result_dict['text'])
            
            return ' '.join(transcript_parts).strip()
            
        except Exception as e:
            logger.error(f"Vosk transcription error: {e}")
            return None
    
    async def _transcribe_google(self, audio_data: np.ndarray, language: str) -> Optional[str]:
        """Transcribe using Google Speech-to-Text"""
        # Placeholder implementation
        logger.warning("Google STT not fully implemented")
        return None
    
    async def transcribe_stream(self, audio_stream, language: str = "en"):
        """Streaming transcription"""
        if self.engines.get(self.active_engine, {}).get('realtime', False):
            return await self._transcribe_stream_realtime(audio_stream, language)
        else:
            return await self._transcribe_stream_batch(audio_stream, language)
    
    async def _transcribe_stream_realtime(self, audio_stream, language: str):
        """Real-time streaming transcription"""
        # This would implement real-time transcription
        # For now, collect all audio and transcribe at once
        
        audio_chunks = []
        async for chunk in audio_stream:
            audio_chunks.append(chunk)
        
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
            return await self.transcribe(audio_data, language)
        
        return None
    
    def get_available_engines(self) -> list:
        """Get list of available STT engines"""
        return list(self.engines.keys())
    
    def set_active_engine(self, engine_name: str) -> bool:
        """Set the active STT engine"""
        if engine_name in self.engines:
            self.active_engine = engine_name
            logger.info(f"Active STT engine set to: {engine_name}")
            return True
        else:
            logger.error(f"STT engine not available: {engine_name}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get STT statistics"""
        stats = self.stats.copy()
        stats['active_engine'] = self.active_engine
        stats['available_engines'] = self.get_available_engines()
        
        if stats['total_transcriptions'] > 0:
            stats['success_rate'] = (
                stats['successful_transcriptions'] / stats['total_transcriptions']
            )
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    async def stop(self):
        """Cleanup resources"""
        # Cleanup engine resources
        for engine_name, engine_info in self.engines.items():
            if hasattr(engine_info.get('engine'), 'close'):
                engine_info['engine'].close()
        
        self.engines.clear()
        logger.info("STT Manager stopped")