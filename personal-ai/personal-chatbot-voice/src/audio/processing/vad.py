"""
Voice Activity Detection using energy-based method
"""

import numpy as np
from typing import List, Tuple, Optional
import sys

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class VoiceActivityDetector:
    """Detects voice activity in audio streams using energy detection"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # VAD parameters
        self.sample_rate = config.get('sample_rate', 16000)
        self.aggressiveness = config.get('aggressiveness', 3)
        self.frame_duration_ms = config.get('frame_duration_ms', 30)
        self.min_speech_duration = config.get('min_speech_duration', 0.5)
        self.min_silence_duration = config.get('min_silence_duration', 0.5)
        self.padding_duration = config.get('padding_duration', 0.1)
        self.threshold = self._get_threshold(config.get('threshold', 0.5))
        
        # Calculate frame size in samples
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        # Ensure frame size is reasonable
        if self.frame_size < 160:  # At least 10ms at 16kHz
            self.frame_size = 160
            self.frame_duration_ms = self.frame_size * 1000 / self.sample_rate
        
    def _get_threshold(self, threshold_config):
        """Convert threshold configuration to actual energy threshold"""
        # Map aggressiveness to threshold
        aggressiveness_map = {
            1: 0.001,  # Very sensitive
            2: 0.005,  # Sensitive
            3: 0.01,   # Moderate (default)
            4: 0.02    # Aggressive
        }
        
        if isinstance(threshold_config, (int, float)):
            return threshold_config
        
        # Use aggressiveness if threshold not specified
        return aggressiveness_map.get(self.aggressiveness, 0.01)
    
    def is_speech(self, audio_frame: np.ndarray) -> bool:
        """Detect if a single frame contains speech using energy"""
        
        if len(audio_frame) == 0:
            return False
        
        # Convert to float for processing
        if audio_frame.dtype != np.float32:
            try:
                audio_float = audio_frame.astype(np.float32) / np.iinfo(audio_frame.dtype).max
            except:
                # If we can't get max, just convert
                audio_float = audio_frame.astype(np.float32) / 32768.0  # Assume 16-bit
        else:
            audio_float = audio_frame
        
        # Ensure correct frame size
        if len(audio_float) != self.frame_size:
            # Resize if necessary
            if len(audio_float) > self.frame_size:
                audio_float = audio_float[:self.frame_size]
            else:
                padding = np.zeros(self.frame_size - len(audio_float), dtype=np.float32)
                audio_float = np.concatenate([audio_float, padding])
        
        # Calculate energy (RMS)
        energy = np.sqrt(np.mean(audio_float ** 2))
        
        # Apply threshold based on aggressiveness
        return energy > self.threshold
    
    def detect_voice_activity(self, audio_data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect voice activity in audio data
        
        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        
        if len(audio_data) == 0:
            return []
        
        # Split audio into frames
        num_frames = len(audio_data) // self.frame_size
        frames = []
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio_data[start:end]
            frames.append(frame)
        
        # Detect speech in each frame
        speech_flags = [self.is_speech(frame) for frame in frames]
        
        # Convert to speech segments
        segments = []
        in_speech = False
        speech_start = 0
        
        min_speech_frames = int(self.min_speech_duration * 1000 / self.frame_duration_ms)
        min_silence_frames = int(self.min_silence_duration * 1000 / self.frame_duration_ms)
        padding_frames = int(self.padding_duration * 1000 / self.frame_duration_ms)
        
        for i, is_speech in enumerate(speech_flags):
            if is_speech and not in_speech:
                # Speech start
                speech_start = i
                in_speech = True
            elif not is_speech and in_speech:
                # Speech end
                speech_duration = i - speech_start
                
                if speech_duration >= min_speech_frames:
                    # Add padding
                    start_frame = max(0, speech_start - padding_frames)
                    end_frame = min(len(speech_flags), i + padding_frames)
                    
                    segments.append((
                        start_frame * self.frame_size,
                        end_frame * self.frame_size
                    ))
                
                in_speech = False
        
        # Handle case where speech continues to end
        if in_speech:
            speech_duration = len(speech_flags) - speech_start
            if speech_duration >= min_speech_frames:
                end_frame = min(len(speech_flags), len(speech_flags) + padding_frames)
                segments.append((
                    speech_start * self.frame_size,
                    end_frame * self.frame_size
                ))
        
        # Merge close segments
        merged_segments = []
        if segments:
            current_start, current_end = segments[0]
            
            for start, end in segments[1:]:
                gap = start - current_end
                max_gap = min_silence_frames * self.frame_size
                
                if gap <= max_gap:
                    # Merge segments
                    current_end = end
                else:
                    # Start new segment
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            # Add last segment
            merged_segments.append((current_start, current_end))
        
        return merged_segments
    
    def extract_speech_segments(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Extract speech segments from audio data"""
        segments = self.detect_voice_activity(audio_data)
        
        speech_segments = []
        for start, end in segments:
            segment = audio_data[start:end]
            if len(segment) > 0:
                speech_segments.append(segment)
        
        return speech_segments
    
    def remove_silence(self, audio_data: np.ndarray) -> np.ndarray:
        """Remove silence from audio data"""
        speech_segments = self.extract_speech_segments(audio_data)
        
        if speech_segments:
            return np.concatenate(speech_segments)
        
        return np.array([], dtype=audio_data.dtype)
    
    def get_speech_ratio(self, audio_data: np.ndarray) -> float:
        """Get ratio of speech to non-speech in audio"""
        segments = self.detect_voice_activity(audio_data)
        
        speech_samples = 0
        for start, end in segments:
            speech_samples += (end - start)
        
        total_samples = len(audio_data)
        
        if total_samples > 0:
            return speech_samples / total_samples
        
        return 0.0
    
    def real_time_vad(self, audio_stream, callback: callable):
        """
        Real-time VAD processing
        
        Args:
            audio_stream: Generator yielding audio chunks
            callback: Function called with (is_speech, audio_chunk)
        """
        buffer = np.array([], dtype=np.int16)
        min_buffer_samples = self.frame_size * 10  # Buffer 10 frames
        
        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk]) if len(buffer) > 0 else chunk
            
            # Process when we have enough data
            while len(buffer) >= self.frame_size:
                frame = buffer[:self.frame_size]
                buffer = buffer[self.frame_size:]
                
                is_speech = self.is_speech(frame)
                callback(is_speech, frame)


# Alternative: Simple energy-based VAD for quick use
class SimpleVAD:
    """Even simpler energy-based VAD"""
    
    def __init__(self, energy_threshold=0.01, sample_rate=16000):
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Simple energy-based speech detection"""
        if len(audio_chunk) == 0:
            return False
        
        # Convert to float
        if audio_chunk.dtype != np.float32:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms > self.energy_threshold
    
    def detect_speech_segments(self, audio_data: np.ndarray, window_ms=30):
        """Detect speech segments"""
        window_size = int(self.sample_rate * window_ms / 1000)
        segments = []
        current_segment = None
        
        for i in range(0, len(audio_data), window_size):
            chunk = audio_data[i:i+window_size]
            
            if self.is_speech(chunk):
                if current_segment is None:
                    current_segment = [i, i + window_size]
                else:
                    current_segment[1] = i + window_size
            elif current_segment is not None:
                segments.append(tuple(current_segment))
                current_segment = None
        
        if current_segment is not None:
            segments.append(tuple(current_segment))
        
        return segments