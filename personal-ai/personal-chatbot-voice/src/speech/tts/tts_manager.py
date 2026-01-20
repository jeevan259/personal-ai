"""
Text-to-Speech Manager supporting multiple engines
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
import numpy as np
from datetime import datetime

from src.utils.logging_config import setup_logging
from src.utils.audio_utils import save_audio_file, load_audio_file

logger = setup_logging(__name__)


class TTSManager:
    """Manages multiple TTS engines with caching and voice management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Available TTS engines
        self.engines = {}
        self.active_engine = config.get('tts_provider', 'elevenlabs')
        self.engine_order = config.get('engine_order', ['elevenlabs', 'openai', 'pyttsx3', 'edge'])
        
        # Voice settings
        self.voice_settings = config.get('voice', {})
        self.current_voice_id = self.voice_settings.get('id', '21m00Tcm4TlvDq8ikWAM')
        
        # Audio cache
        self.cache_enabled = config.get('caching', {}).get('enabled', True)
        self.cache_dir = Path("data/tts_cache")
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'total_synthesies': 0,
            'cache_hits': 0,
            'average_latency': 0.0
        }
        
        # Voice cloning
        self.voice_cloning_enabled = config.get('voice_cloning', {}).get('enabled', False)
        self.voice_cloner = None
        
    async def initialize(self):
        """Initialize TTS engines"""
        logger.info("Initializing TTS Manager...")
        
        try:
            # Initialize engines in order
            for engine_name in self.engine_order:
                if await self._initialize_engine(engine_name):
                    logger.info(f"Initialized {engine_name} TTS engine")
                else:
                    logger.warning(f"Failed to initialize {engine_name} TTS engine")
            
            # Check if active engine was initialized
            if self.active_engine not in self.engines:
                # Fallback to first available engine
                available_engines = list(self.engines.keys())
                if available_engines:
                    self.active_engine = available_engines[0]
                    logger.info(f"Falling back to {self.active_engine} as active engine")
                else:
                    raise RuntimeError("No TTS engines could be initialized")
            
            # Initialize voice cloning if enabled
            if self.voice_cloning_enabled:
                await self._initialize_voice_cloning()
            
            logger.info(f"TTS Manager initialized with active engine: {self.active_engine}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS Manager: {e}", exc_info=True)
            return False
    
    async def _initialize_engine(self, engine_name: str) -> bool:
        """Initialize a specific TTS engine"""
        try:
            if engine_name == 'elevenlabs':
                return await self._init_elevenlabs()
            elif engine_name == 'openai':
                return await self._init_openai_tts()
            elif engine_name == 'pyttsx3':
                return await self._init_pyttsx3()
            elif engine_name == 'edge':
                return await self._init_edge_tts()
            else:
                logger.warning(f"Unknown TTS engine: {engine_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing {engine_name}: {e}")
            return False
    
    async def _init_elevenlabs(self) -> bool:
        """Initialize ElevenLabs TTS"""
        try:
            from elevenlabs import ElevenLabs
            
            api_key = self.config.get('elevenlabs_api_key')
            if not api_key:
                logger.warning("ElevenLabs API key not provided")
                return False
            
            client = ElevenLabs(api_key=api_key)
            
            self.engines['elevenlabs'] = {
                'client': client,
                'type': 'elevenlabs',
                'voices': {},
                'streaming': True,
                'quality': 'high'
            }
            
            # Load available voices
            await self._load_elevenlabs_voices(client)
            
            return True
            
        except ImportError:
            logger.warning("ElevenLabs not installed. Install with: pip install elevenlabs")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs: {e}")
            return False
    
    async def _load_elevenlabs_voices(self, client):
        """Load available ElevenLabs voices"""
        try:
            voices = client.voices.get_all()
            self.engines['elevenlabs']['voices'] = {
                voice.voice_id: {
                    'name': voice.name,
                    'category': voice.category,
                    'description': voice.description,
                    'preview_url': voice.preview_url
                }
                for voice in voices.voices
            }
            
            logger.info(f"Loaded {len(voices.voices)} ElevenLabs voices")
            
        except Exception as e:
            logger.error(f"Failed to load ElevenLabs voices: {e}")
    
    async def _init_openai_tts(self) -> bool:
        """Initialize OpenAI TTS"""
        try:
            from openai import OpenAI
            
            api_key = self.config.get('openai_api_key')
            if not api_key:
                logger.warning("OpenAI API key not provided")
                return False
            
            client = OpenAI(api_key=api_key)
            
            self.engines['openai'] = {
                'client': client,
                'type': 'openai',
                'voices': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
                'streaming': True,
                'quality': 'medium'
            }
            
            return True
            
        except ImportError:
            logger.warning("OpenAI not installed. Install with: pip install openai")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI TTS: {e}")
            return False
    
    async def _init_pyttsx3(self) -> bool:
        """Initialize pyttsx3 (offline TTS)"""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Configure engine
            voices = engine.getProperty('voices')
            voice_dict = {}
            
            for voice in voices:
                voice_dict[voice.id] = {
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': voice.gender
                }
            
            self.engines['pyttsx3'] = {
                'engine': engine,
                'type': 'pyttsx3',
                'voices': voice_dict,
                'streaming': False,
                'quality': 'low'
            }
            
            return True
            
        except ImportError:
            logger.warning("pyttsx3 not installed. Install with: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            return False
    
    async def _init_edge_tts(self) -> bool:
        """Initialize Edge TTS"""
        try:
            import edge_tts
            
            self.engines['edge'] = {
                'type': 'edge',
                'voices': {},  # Will be loaded on demand
                'streaming': True,
                'quality': 'medium'
            }
            
            return True
            
        except ImportError:
            logger.warning("edge-tts not installed. Install with: pip install edge-tts")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Edge TTS: {e}")
            return False
    
    async def _initialize_voice_cloning(self):
        """Initialize voice cloning"""
        try:
            from src.speech.voice_cloning.voice_manager import VoiceCloningManager
            
            voice_config = self.config.get('voice_cloning', {})
            self.voice_cloner = VoiceCloningManager(voice_config)
            await self.voice_cloner.initialize()
            
            logger.info("Voice cloning initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize voice cloning: {e}")
            self.voice_cloning_enabled = False
    
    def _get_cache_key(self, text: str, voice_id: str, engine: str) -> str:
        """Generate cache key for TTS output"""
        # Create a hash of text + voice + engine + voice settings
        cache_string = f"{text}_{voice_id}_{engine}_{json.dumps(self.voice_settings, sort_keys=True)}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Get audio from cache"""
        if not self.cache_enabled:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.mp3"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
                
                # Update cache metadata
                metadata_file = self.cache_dir / f"{cache_key}.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    metadata['last_accessed'] = datetime.now().isoformat()
                    metadata['access_count'] = metadata.get('access_count', 0) + 1
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f)
                
                self.stats['cache_hits'] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return audio_data
                
            except Exception as e:
                logger.error(f"Error reading from cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, audio_data: bytes, metadata: Dict[str, Any]):
        """Save audio to cache"""
        if not self.cache_enabled:
            return
        
        try:
            # Save audio file
            audio_file = self.cache_dir / f"{cache_key}.mp3"
            with open(audio_file, 'wb') as f:
                f.write(audio_data)
            
            # Save metadata
            metadata_file = self.cache_dir / f"{cache_key}.json"
            metadata['created'] = datetime.now().isoformat()
            metadata['text_length'] = len(metadata.get('text', ''))
            metadata['voice_id'] = metadata.get('voice_id', self.current_voice_id)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            logger.debug(f"Saved to cache: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    async def synthesize(
        self,
        text: str,
        voice_settings: Optional[Dict[str, Any]] = None,
        engine: Optional[str] = None,
        streaming: bool = False
    ) -> Optional[bytes]:
        """Synthesize text to speech"""
        
        if engine is None:
            engine = self.active_engine
        
        if engine not in self.engines:
            logger.error(f"TTS engine not available: {engine}")
            # Try fallback
            for fallback_engine in self.engine_order:
                if fallback_engine in self.engines and fallback_engine != engine:
                    logger.info(f"Falling back to {fallback_engine}")
                    return await self.synthesize(text, voice_settings, fallback_engine, streaming)
            return None
        
        # Merge voice settings
        settings = self.voice_settings.copy()
        if voice_settings:
            settings.update(voice_settings)
        
        # Check cache first
        cache_key = self._get_cache_key(text, settings.get('id', self.current_voice_id), engine)
        cached_audio = self._get_cached_audio(cache_key)
        
        if cached_audio and not streaming:
            return cached_audio
        
        try:
            # Start timing
            import time
            start_time = time.time()
            
            # Perform synthesis
            if engine == 'elevenlabs':
                audio_data = await self._synthesize_elevenlabs(text, settings, streaming)
            elif engine == 'openai':
                audio_data = await self._synthesize_openai(text, settings, streaming)
            elif engine == 'pyttsx3':
                audio_data = await self._synthesize_pyttsx3(text, settings)
            elif engine == 'edge':
                audio_data = await self._synthesize_edge(text, settings, streaming)
            else:
                logger.error(f"Unsupported engine: {engine}")
                return None
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Update statistics
            self.stats['total_synthesies'] += 1
            
            if audio_data:
                # Update average latency
                alpha = 0.1
                self.stats['average_latency'] = (
                    alpha * latency + 
                    (1 - alpha) * self.stats['average_latency']
                )
                
                # Save to cache if not streaming
                if not streaming and audio_data:
                    metadata = {
                        'text': text,
                        'voice_id': settings.get('id', self.current_voice_id),
                        'engine': engine,
                        'latency': latency,
                        'text_hash': cache_key
                    }
                    self._save_to_cache(cache_key, audio_data, metadata)
            
            logger.debug(f"TTS ({engine}) latency: {latency:.3f}s")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Synthesis failed with {engine}: {e}", exc_info=True)
            
            # Try fallback engine
            if engine != self.engine_order[0]:
                for fallback_engine in self.engine_order:
                    if (fallback_engine in self.engines and 
                        fallback_engine != engine and 
                        fallback_engine != self.active_engine):
                        logger.info(f"Trying fallback engine: {fallback_engine}")
                        return await self.synthesize(text, voice_settings, fallback_engine, streaming)
            
            return None
    
    async def _synthesize_elevenlabs(
        self, 
        text: str, 
        settings: Dict[str, Any], 
        streaming: bool = False
    ) -> Optional[bytes]:
        """Synthesize using ElevenLabs"""
        try:
            engine_info = self.engines['elevenlabs']
            client = engine_info['client']
            
            voice_id = settings.get('id', self.current_voice_id)
            
            # Prepare generation parameters
            generation_params = {
                'voice': voice_id,
                'model': 'eleven_monolingual_v1',
                'output_format': 'mp3_22050_32',
            }
            
            # Add voice settings if provided
            if 'stability' in settings:
                generation_params['voice_settings'] = {
                    'stability': settings.get('stability', 0.5),
                    'similarity_boost': settings.get('similarity_boost', 0.75)
                }
            
            if streaming:
                # Generate streaming audio
                audio_stream = client.generate(
                    text=text,
                    stream=True,
                    **generation_params
                )
                
                # Collect stream data
                audio_chunks = []
                for chunk in audio_stream:
                    audio_chunks.append(chunk)
                
                audio_data = b''.join(audio_chunks)
            else:
                # Generate audio directly
                audio_data = client.generate(
                    text=text,
                    **generation_params
                )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"ElevenLabs synthesis error: {e}")
            return None
    
    async def _synthesize_openai(
        self, 
        text: str, 
        settings: Dict[str, Any], 
        streaming: bool = False
    ) -> Optional[bytes]:
        """Synthesize using OpenAI TTS"""
        try:
            engine_info = self.engines['openai']
            client = engine_info['client']
            
            voice = settings.get('voice', 'nova')
            
            if streaming:
                # Streaming not fully supported in current OpenAI API
                # Fallback to non-streaming
                streaming = False
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                response_format="mp3"
            )
            
            # Read the audio data
            audio_data = response.content
            
            return audio_data
            
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}")
            return None
    
    async def _synthesize_pyttsx3(self, text: str, settings: Dict[str, Any]) -> Optional[bytes]:
        """Synthesize using pyttsx3"""
        try:
            engine_info = self.engines['pyttsx3']
            engine = engine_info['engine']
            
            # Configure voice
            voice_id = settings.get('id')
            if voice_id and voice_id in engine_info['voices']:
                engine.setProperty('voice', voice_id)
            
            # Configure speech rate
            rate = settings.get('rate', 150)
            engine.setProperty('rate', rate)
            
            # Configure volume
            volume = settings.get('volume', 1.0)
            engine.setProperty('volume', volume)
            
            # Save to temporary file
            import tempfile
            import wave
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpfile:
                tmp_path = tmpfile.name
            
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            
            # Read the generated file
            with open(tmp_path, 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            import os
            os.unlink(tmp_path)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis error: {e}")
            return None
    
    async def _synthesize_edge(
        self, 
        text: str, 
        settings: Dict[str, Any], 
        streaming: bool = False
    ) -> Optional[bytes]:
        """Synthesize using Edge TTS"""
        try:
            import edge_tts
            
            voice = settings.get('voice', 'en-US-JennyNeural')
            
            # Create communicate object
            communicate = edge_tts.Communicate(text, voice)
            
            # Generate audio
            audio_chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_chunks.append(chunk["data"])
            
            audio_data = b''.join(audio_chunks)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Edge TTS synthesis error: {e}")
            return None
    
    async def set_voice_settings(self, voice_settings: Dict[str, Any]):
        """Update voice settings"""
        self.voice_settings.update(voice_settings)
        
        if 'id' in voice_settings:
            self.current_voice_id = voice_settings['id']
            logger.info(f"Voice set to: {self.current_voice_id}")
    
    async def clone_voice(self, reference_audio: bytes, voice_name: str) -> Optional[str]:
        """Clone a voice from reference audio"""
        if not self.voice_cloning_enabled or not self.voice_cloner:
            logger.warning("Voice cloning not enabled or initialized")
            return None
        
        try:
            voice_id = await self.voice_cloner.clone_voice(reference_audio, voice_name)
            return voice_id
            
        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return None
    
    def get_available_voices(self, engine: Optional[str] = None) -> Dict[str, Any]:
        """Get available voices for an engine"""
        if engine is None:
            engine = self.active_engine
        
        if engine in self.engines:
            return self.engines[engine].get('voices', {})
        
        return {}
    
    def get_available_engines(self) -> list:
        """Get list of available TTS engines"""
        return list(self.engines.keys())
    
    def set_active_engine(self, engine_name: str) -> bool:
        """Set the active TTS engine"""
        if engine_name in self.engines:
            self.active_engine = engine_name
            logger.info(f"Active TTS engine set to: {engine_name}")
            return True
        else:
            logger.error(f"TTS engine not available: {engine_name}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        stats = self.stats.copy()
        stats['active_engine'] = self.active_engine
        stats['available_engines'] = self.get_available_engines()
        stats['current_voice'] = self.current_voice_id
        
        if stats['total_synthesies'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_synthesies']
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    async def stop(self):
        """Cleanup resources"""
        # Cleanup engine resources
        for engine_name, engine_info in self.engines.items():
            if engine_name == 'pyttsx3' and 'engine' in engine_info:
                engine_info['engine'].stop()
        
        if self.voice_cloner:
            await self.voice_cloner.stop()
        
        self.engines.clear()
        logger.info("TTS Manager stopped")