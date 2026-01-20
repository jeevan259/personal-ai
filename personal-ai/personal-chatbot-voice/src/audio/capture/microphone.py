"""
Microphone audio capture using PyAudio
"""

import asyncio
import queue
import threading
from typing import Optional, Callable, Any, AsyncGenerator
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not installed. Run: pip install pyaudio")

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class MicrophoneCapture:
    """Captures audio from microphone using PyAudio"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Audio parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_size = config.get('chunk_size', 1024)
        self.channels = config.get('channels', 1)
        self.format = self._get_format(config.get('format', 'int16'))
        self.device_index = config.get('device_index')
        
        # PyAudio objects
        self.pyaudio = None
        self.stream = None
        
        # State
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.callback = None
        
        # Threading
        self.capture_thread = None
        
    def _get_format(self, format_str: str) -> int:
        """Convert format string to PyAudio format constant"""
        if not PYAUDIO_AVAILABLE:
            return None
            
        format_map = {
            'int8': pyaudio.paInt8,
            'int16': pyaudio.paInt16,
            'int24': pyaudio.paInt24,
            'int32': pyaudio.paInt32,
            'float32': pyaudio.paFloat32
        }
        return format_map.get(format_str, pyaudio.paInt16)
    
    async def initialize(self):
        """Initialize the microphone capture"""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available. Install with: pip install pyaudio")
            return False
            
        try:
            self.pyaudio = pyaudio.PyAudio()
            
            # Get device info
            device_info = None
            if self.device_index is not None:
                device_info = self.pyaudio.get_device_info_by_index(self.device_index)
                logger.info(f"Using audio device: {device_info['name']}")
            else:
                # Use default device
                device_info = self.pyaudio.get_default_input_device_info()
                logger.info(f"Using default audio device: {device_info['name']}")
            
            # Validate device supports our parameters
            if device_info['maxInputChannels'] < self.channels:
                logger.warning(f"Device only has {device_info['maxInputChannels']} input channels, "
                              f"requested {self.channels}")
                self.channels = device_info['maxInputChannels']
            
            self.sample_rate = int(device_info['defaultSampleRate'])
            logger.info(f"Sample rate: {self.sample_rate} Hz")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            return False
    
    def start_recording(self, callback: Optional[Callable] = None):
        """Start recording audio"""
        if not PYAUDIO_AVAILABLE:
            logger.error("PyAudio not available")
            return
            
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        if not self.pyaudio:
            logger.error("PyAudio not initialized")
            return
        
        try:
            self.callback = callback
            self.is_recording = True
            
            # Open stream
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._stream_callback if callback else None
            )
            
            if not callback:
                # Start capture thread for synchronous reading
                self.capture_thread = threading.Thread(
                    target=self._capture_loop,
                    daemon=True
                )
                self.capture_thread.start()
            
            self.stream.start_stream()
            logger.info("Started audio recording")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
    
    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        """Callback for async stream"""
        if self.callback:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.callback(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def _capture_loop(self):
        """Capture loop for synchronous reading"""
        while self.is_recording and self.stream and self.stream.is_active():
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                self.audio_queue.put(audio_data)
            except Exception as e:
                if self.is_recording:  # Only log if we're supposed to be recording
                    logger.error(f"Error in capture loop: {e}")
                break
    
    def read_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read a chunk of audio data"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    async def get_audio_stream(self) -> AsyncGenerator[np.ndarray, None]:
        """Async generator yielding audio chunks"""
        while self.is_recording:
            chunk = self.read_chunk()
            if chunk is not None:
                yield chunk
            else:
                await asyncio.sleep(0.01)
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        
        logger.info("Stopped audio recording")
    
    async def record_until_silence(
        self,
        silence_threshold: int = 500,
        silence_duration: float = 1.0,
        max_duration: float = 30.0
    ) -> np.ndarray:
        """Record audio until silence is detected"""
        import time
        from collections import deque
        
        audio_chunks = []
        silence_buffer = deque(maxlen=int(silence_duration * self.sample_rate / self.chunk_size))
        start_time = time.time()
        
        self.start_recording()
        
        try:
            while time.time() - start_time < max_duration:
                chunk = self.read_chunk(timeout=0.1)
                if chunk is None:
                    continue
                
                audio_chunks.append(chunk)
                
                # Check for silence
                rms = np.sqrt(np.mean(chunk.astype(float) ** 2))
                is_silent = rms < silence_threshold
                silence_buffer.append(is_silent)
                
                # If all recent chunks are silent, stop recording
                if len(silence_buffer) == silence_buffer.maxlen and all(silence_buffer):
                    logger.debug("Silence detected, stopping recording")
                    break
                    
                await asyncio.sleep(0.01)
            
        finally:
            self.stop_recording()
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        return np.array([], dtype=np.int16)
    
    def list_devices(self):
        """List available audio devices"""
        if not PYAUDIO_AVAILABLE:
            return []
            
        if not self.pyaudio:
            self.pyaudio = pyaudio.PyAudio()
        
        devices = []
        for i in range(self.pyaudio.get_device_count()):
            dev_info = self.pyaudio.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': dev_info['name'],
                    'channels': dev_info['maxInputChannels'],
                    'sample_rate': dev_info['defaultSampleRate']
                })
        
        return devices
    
    async def stop(self):
        """Cleanup resources"""
        self.stop_recording()
        
        if self.pyaudio:
            self.pyaudio.terminate()
            self.pyaudio = None