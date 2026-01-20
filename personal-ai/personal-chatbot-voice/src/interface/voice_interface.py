"""
Voice interface for interactive voice conversations
"""

import asyncio
import time
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import numpy as np  # ADD THIS IMPORT

from src.core.voice_engine import VoiceEngine
from src.audio.capture.microphone import MicrophoneCapture
from src.audio.output.player import AudioPlayer
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class VoiceInterface:
    """Interactive voice interface for the chatbot"""
    class VoiceInterface:
    
    
     def __init__(self, engine):
        self.engine = engine
        print("✅ VoiceInterface created")
    
    async def start_listening(self):
        """Start audio listening (placeholder)"""
        print("\n🎤 Audio mode placeholder")
        print("-" * 40)
        print("Audio recording would start here.")
        print("\nTo enable real audio, install:")
        print("  pip install pyaudio soundfile openai-whisper")
        print("\nFor now, please use text mode.")
    
    async def record_audio(self):
        """Record audio from microphone"""
        print("Audio recording not yet implemented.")
        print("Use 'text' command instead for now.")
    
    def __init__(self, voice_engine: VoiceEngine):
        self.voice_engine = voice_engine
        self.config = voice_engine.config
        
        # Audio components
        self.microphone: Optional[MicrophoneCapture] = None
        self.audio_player: Optional[AudioPlayer] = None
        
        # State
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        self.last_wake_time = 0
        self.wake_word_cooldown = 2.0
        
        # Callbacks
        self.on_wake_detected: Optional[Callable] = None
        self.on_response_start: Optional[Callable] = None
        self.on_response_end: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        # Conversation state
        self.conversation_active = False
        self.conversation_start_time = 0
        self.conversation_timeout = 30  # seconds
        
    async def initialize(self):
        """Initialize the voice interface"""
        logger.info("Initializing Voice Interface...")
        
        try:
            # Initialize microphone
            self.microphone = MicrophoneCapture(self.config['audio'])
            await self.microphone.initialize()
            
            # Initialize audio player
            self.audio_player = AudioPlayer(self.config['audio'])
            await self.audio_player.initialize()
            
            # Setup wake word detection
            await self._setup_wake_word_detection()
            
            logger.info("Voice Interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Voice Interface: {e}", exc_info=True)
            return False
    
    async def _setup_wake_word_detection(self):
        """Setup wake word detection callbacks"""
        if self.voice_engine.wake_manager:
            await self.voice_engine.wake_manager.start_detection(
                on_wake_detected=self._on_wake_detected,
                on_error=self._on_wake_error
            )
    
    async def _on_wake_detected(self, detection: Dict[str, Any]):
        """Handle wake word detection"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_wake_time < self.wake_word_cooldown:
            logger.debug("Wake word cooldown active")
            return
        
        self.last_wake_time = current_time
        
        logger.info(f"Wake word detected: {detection.get('keyword', 'Unknown')}")
        
        # Play wake sound
        await self._play_wake_sound()
        
        # Start listening for command
        await self.start_listening()
        
        # Call external callback if set
        if self.on_wake_detected:
            await self.on_wake_detected(detection)
    
    async def _on_wake_error(self, error: Exception):
        """Handle wake word detection error"""
        logger.error(f"Wake word detection error: {error}")
        
        if self.on_error:
            await self.on_error(error)
    
    async def _play_wake_sound(self):
        """Play wake sound indication"""
        try:
            # Generate a simple wake tone
            sample_rate = 24000
            duration = 0.2
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create a pleasant chime
            freq1 = 800
            freq2 = 1200
            tone = 0.3 * (np.sin(2 * np.pi * freq1 * t) + 
                         np.sin(2 * np.pi * freq2 * t) * np.exp(-5 * t))
            
            # Apply fade
            fade_samples = int(0.05 * sample_rate)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            
            if len(tone) > 2 * fade_samples:
                tone[:fade_samples] *= fade_in
                tone[-fade_samples:] *= fade_out
            
            # Play the tone
            await self.audio_player.play_audio(tone, sample_rate)
            
        except Exception as e:
            logger.error(f"Error playing wake sound: {e}")
    
    async def start_listening(self):
        """Start listening for voice command"""
        if self.is_listening or self.is_processing:
            return
        
        self.is_listening = True
        self.conversation_active = True
        self.conversation_start_time = time.time()
        
        logger.info("Listening for voice command...")
        
        # Play listening indicator
        await self._play_listening_indicator()
        
        # Start recording
        await self._record_and_process()
    
    async def _play_listening_indicator(self):
        """Play a sound to indicate listening"""
        try:
            # Simple beep to indicate listening
            sample_rate = 24000
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = 0.1 * np.sin(2 * np.pi * 1000 * t)
            
            await self.audio_player.play_audio(beep, sample_rate)
            
        except Exception as e:
            logger.debug(f"Could not play listening indicator: {e}")
    
    async def _record_and_process(self):
        """Record audio and process it"""
        if not self.microphone:
            logger.error("Microphone not initialized")
            return
        
        try:
            # Record audio until silence
            audio_data = await self.microphone.record_until_silence(
                silence_threshold=self.config['audio'].get('silence_threshold', 500),
                silence_duration=self.config['audio'].get('silence_duration', 1.0),
                max_duration=self.config['audio'].get('max_record_seconds', 30)
            )
            
            if len(audio_data) == 0:
                logger.warning("No audio recorded")
                await self.stop_listening()
                return
            
            # Process the audio
            await self.process_audio(audio_data)
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}", exc_info=True)
            await self.stop_listening()
    
    async def process_audio(self, audio_data):
        """Process recorded audio"""
        if self.is_processing:
            logger.warning("Already processing audio")
            return
        
        self.is_processing = True
        self.is_listening = False
        
        try:
            logger.info("Processing audio...")
            
            # Process through voice engine
            response = await self.voice_engine.process_audio_stream([audio_data])
            
            if response:
                await self._handle_response(response)
            else:
                logger.warning("No response generated")
                await self._play_error_sound()
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            await self._play_error_sound()
            
        finally:
            self.is_processing = False
            
            # Check if conversation should continue
            if self.conversation_active:
                # Continue listening for follow-up
                await asyncio.sleep(0.5)
                await self.start_listening()
    
    async def _handle_response(self, response: Dict[str, Any]):
        """Handle response from voice engine"""
        
        response_type = response.get('type', 'response')
        
        if response_type == 'interruption':
            logger.info("Interruption handled")
            await self._play_interruption_sound()
            return
        
        # Get response text and audio
        response_text = response.get('text', '')
        response_audio = response.get('audio')
        
        if not response_text:
            logger.warning("Empty response text")
            return
        
        logger.info(f"Assistant: {response_text}")
        
        # Call response start callback
        if self.on_response_start:
            await self.on_response_start(response_text)
        
        # Speak the response
        if response_audio:
            await self.speak_audio(response_audio)
        else:
            # Generate TTS for response text
            await self.speak_text(response_text)
        
        # Call response end callback
        if self.on_response_end:
            await self.on_response_end(response_text)
    
    async def speak_text(self, text: str, voice_settings: Optional[Dict[str, Any]] = None):
        """Speak text using TTS"""
        if self.is_speaking:
            logger.warning("Already speaking")
            return
        
        self.is_speaking = True
        
        try:
            # Generate speech
            audio_data = await self.voice_engine.tts_manager.synthesize(
                text,
                voice_settings=voice_settings
            )
            
            if audio_data:
                await self.speak_audio(audio_data)
            else:
                logger.error("Failed to generate speech from text")
                
        except Exception as e:
            logger.error(f"Error speaking text: {e}", exc_info=True)
            
        finally:
            self.is_speaking = False
    
    async def speak_audio(self, audio_data: bytes):
        """Play audio data"""
        if not self.audio_player:
            logger.error("Audio player not initialized")
            return
        
        try:
            # Try to load as MP3
            try:
                import soundfile as sf
                import io
                
                audio, sample_rate = sf.read(io.BytesIO(audio_data))
                
            except:
                # Try as WAV
                try:
                    import wave
                    import io as bio
                    
                    with wave.open(bio.BytesIO(audio_data)) as wav_file:
                        sample_rate = wav_file.getframerate()
                        audio = np.frombuffer(
                            wav_file.readframes(wav_file.getnframes()),
                            dtype=np.int16
                        )
                except:
                    # Fallback: assume raw PCM data
                    sample_rate = 24000
                    audio = np.frombuffer(audio_data, dtype=np.int16)
            
            # Play audio
            await self.audio_player.play_audio(audio, sample_rate)
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)
    
    async def _play_error_sound(self):
        """Play error sound"""
        try:
            sample_rate = 24000
            duration = 0.3
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create error tone
            error_tone = 0.2 * np.sin(2 * np.pi * 300 * t) * np.exp(-3 * t)
            
            await self.audio_player.play_audio(error_tone, sample_rate)
            
        except Exception as e:
            logger.debug(f"Could not play error sound: {e}")
    
    async def _play_interruption_sound(self):
        """Play interruption sound"""
        try:
            sample_rate = 24000
            duration = 0.1
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Short beep for interruption
            beep = 0.1 * np.sin(2 * np.pi * 800 * t)
            
            await self.audio_player.play_audio(beep, sample_rate)
            
        except Exception as e:
            logger.debug(f"Could not play interruption sound: {e}")
    
    async def stop_listening(self):
        """Stop listening for voice commands"""
        if not self.is_listening:
            return
        
        self.is_listening = False
        self.conversation_active = False
        
        logger.info("Stopped listening")
    
    async def process_voice_command(self, audio_file: Optional[Path] = None):
        """Process a voice command from file or microphone"""
        try:
            if audio_file:
                # Process audio file
                from src.utils.audio_utils import load_audio_file
                
                audio_data, sample_rate = load_audio_file(audio_file)
                logger.info(f"Processing audio file: {audio_file}")
                
                # Resample if needed
                target_rate = self.config['audio'].get('sample_rate', 16000)
                if sample_rate != target_rate:
                    from src.utils.audio_utils import resample_audio
                    audio_data = resample_audio(audio_data, sample_rate, target_rate)
                
                await self.process_audio(audio_data)
                
            else:
                # Process from microphone
                await self.start_listening()
                
        except Exception as e:
            logger.error(f"Error processing voice command: {e}", exc_info=True)
    
    async def listen_for_wake_word(self) -> bool:
        """Listen for wake word (non-blocking)"""
        # This would typically be handled by the wake word manager
        # For this interface, we'll simulate it
        
        if not self.conversation_active and self.voice_engine.wake_manager:
            # Wake word detection is running in background
            # Check if we should start listening
            return False
        
        return False
    
    async def stop(self):
        """Stop the voice interface"""
        logger.info("Stopping Voice Interface...")
        
        # Stop listening
        await self.stop_listening()
        
        # Stop wake word detection
        if self.voice_engine.wake_manager:
            await self.voice_engine.wake_manager.stop_detection()
        
        # Stop audio components
        if self.microphone:
            await self.microphone.stop()
        
        if self.audio_player:
            await self.audio_player.stop()
        
        logger.info("Voice Interface stopped")