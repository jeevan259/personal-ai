"""
Audio utility functions for processing and conversion
WITH safe fallbacks for missing dependencies
"""

import numpy as np
from typing import Union, Optional, Tuple
from pathlib import Path
import wave
import io
import sys


# Check for optional dependencies at import time
try:
    import resampy
    RESAMPY_AVAILABLE = True
except ImportError:
    RESAMPY_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pyloudnorm as pyln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def normalize_audio(audio_data: np.ndarray, target_level: float = 0.1) -> np.ndarray:
    """
    Normalize audio to target level
    
    Args:
        audio_data: Input audio array
        target_level: Target peak level (0.0 to 1.0)
    
    Returns:
        Normalized audio array
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Ensure audio is float
    if audio_data.dtype != np.float32:
        audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    else:
        audio_float = audio_data.copy()
    
    # Calculate current peak
    current_peak = np.max(np.abs(audio_float))
    
    if current_peak > 0:
        # Normalize to target level
        gain = target_level / current_peak
        normalized = audio_float * gain
        
        # Apply gentle limiting to prevent clipping
        clip_threshold = 0.99
        if np.max(np.abs(normalized)) > clip_threshold:
            normalized = np.tanh(normalized / clip_threshold) * clip_threshold
        
        return normalized
    
    return audio_float


def resample_audio(
    audio_data: np.ndarray,
    original_rate: int,
    target_rate: int,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Resample audio to target sample rate
    
    Args:
        audio_data: Input audio array
        original_rate: Original sample rate
        target_rate: Target sample rate
        dtype: Output data type
    
    Returns:
        Resampled audio array
    """
    if original_rate == target_rate:
        return audio_data.astype(dtype)
    
    # Try using resampy if available
    if RESAMPY_AVAILABLE:
        try:
            # Convert to float for resampling
            if audio_data.dtype != np.float32:
                if np.issubdtype(audio_data.dtype, np.integer):
                    audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                else:
                    audio_float = audio_data.astype(np.float32)
            else:
                audio_float = audio_data
            
            # Resample
            resampled = resampy.resample(audio_float, original_rate, target_rate, filter='kaiser_best')
            
            # Convert back to target dtype
            if dtype == np.float32:
                return resampled
            elif np.issubdtype(dtype, np.integer):
                return (resampled * np.iinfo(dtype).max).astype(dtype)
            else:
                return resampled.astype(dtype)
                
        except Exception as e:
            print(f"Resampy resampling failed: {e}. Using fallback.")
    
    # Fallback: simple linear interpolation
    ratio = target_rate / original_rate
    new_length = int(len(audio_data) * ratio)
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(audio_data))
    x_new = np.linspace(0, 1, new_length)
    
    resampled = np.interp(x_new, x_old, audio_data)
    
    return resampled.astype(dtype)


def remove_noise(
    audio_data: np.ndarray,
    sample_rate: int,
    noise_reduction_db: float = 20
) -> np.ndarray:
    """
    Apply noise reduction to audio
    
    Args:
        audio_data: Input audio array
        sample_rate: Audio sample rate
        noise_reduction_db: Noise reduction in dB
    
    Returns:
        Noise-reduced audio array
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Try using noisereduce if available
    if NOISEREDUCE_AVAILABLE:
        try:
            # Ensure audio is float
            if audio_data.dtype != np.float32:
                audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            else:
                audio_float = audio_data
            
            # Estimate noise from first 100ms
            noise_samples = int(sample_rate * 0.1)
            if len(audio_float) > noise_samples:
                noise_sample = audio_float[:noise_samples]
            else:
                noise_sample = audio_float
            
            # Apply noise reduction
            reduced = nr.reduce_noise(
                y=audio_float,
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=noise_reduction_db / 100.0,
                n_fft=1024,
                win_length=1024,
                hop_length=256
            )
            
            return reduced
            
        except Exception as e:
            print(f"Noise reduction failed: {e}. Using fallback.")
    
    # Fallback: simple high-pass filter using numpy
    if SCIPY_AVAILABLE:
        try:
            # Design high-pass filter
            nyquist = sample_rate / 2
            cutoff = 80  # Hz
            b, a = signal.butter(4, cutoff / nyquist, btype='high')
            
            # Apply filter
            filtered = signal.filtfilt(b, a, audio_data)
            
            return filtered
        except:
            pass
    
    # Ultimate fallback: just return the original
    return audio_data.copy()


def detect_silence(
    audio_data: np.ndarray,
    threshold: float = 0.01,
    min_silence_duration: float = 0.1,
    sample_rate: int = 16000
) -> np.ndarray:
    """
    Detect silence in audio
    
    Args:
        audio_data: Input audio array
        threshold: Silence threshold
        min_silence_duration: Minimum silence duration in seconds
        sample_rate: Audio sample rate
    
    Returns:
        Boolean array indicating silence
    """
    if len(audio_data) == 0:
        return np.array([], dtype=bool)
    
    # Calculate energy
    if audio_data.dtype != np.float32:
        audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    else:
        audio_float = audio_data
    
    # Calculate RMS in windows
    window_size = int(sample_rate * 0.01)  # 10ms windows
    if len(audio_float) < window_size:
        return np.zeros(len(audio_float), dtype=bool)
    
    rms = np.sqrt(np.convolve(audio_float ** 2, np.ones(window_size) / window_size, mode='same'))
    
    # Detect silence
    is_silent = rms < threshold
    
    # Apply minimum silence duration
    min_silence_samples = int(sample_rate * min_silence_duration)
    
    # Find silent regions
    silent_regions = []
    in_silence = False
    start_idx = 0
    
    for i in range(len(is_silent)):
        if is_silent[i] and not in_silence:
            start_idx = i
            in_silence = True
        elif not is_silent[i] and in_silence:
            if i - start_idx >= min_silence_samples:
                silent_regions.append((start_idx, i))
            in_silence = False
    
    # Handle ending in silence
    if in_silence and len(is_silent) - start_idx >= min_silence_samples:
        silent_regions.append((start_idx, len(is_silent)))
    
    # Create output array
    silence_mask = np.zeros(len(audio_float), dtype=bool)
    
    for start, end in silent_regions:
        silence_mask[start:end] = True
    
    return silence_mask


def split_on_silence(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    min_silence_len: float = 0.5,
    silence_thresh: float = 0.01,
    keep_silence: float = 0.1
) -> list:
    """
    Split audio on silence
    
    Args:
        audio_data: Input audio array
        sample_rate: Audio sample rate
        min_silence_len: Minimum silence length to split on (seconds)
        silence_thresh: Silence threshold
        keep_silence: Silence to keep at edges (seconds)
    
    Returns:
        List of audio segments
    """
    if len(audio_data) == 0:
        return []
    
    # Detect silence
    silence_mask = detect_silence(
        audio_data,
        threshold=silence_thresh,
        min_silence_duration=min_silence_len,
        sample_rate=sample_rate
    )
    
    # Find non-silent segments
    segments = []
    in_speech = False
    start_idx = 0
    
    for i in range(len(silence_mask)):
        if not silence_mask[i] and not in_speech:
            start_idx = i
            in_speech = True
        elif silence_mask[i] and in_speech:
            # Add keep_silence padding
            pad_samples = int(sample_rate * keep_silence)
            segment_start = max(0, start_idx - pad_samples)
            segment_end = min(len(audio_data), i + pad_samples)
            
            segments.append(audio_data[segment_start:segment_end])
            in_speech = False
    
    # Handle ending in speech
    if in_speech:
        pad_samples = int(sample_rate * keep_silence)
        segment_start = max(0, start_idx - pad_samples)
        segments.append(audio_data[segment_start:])
    
    return segments


def calculate_rms(audio_data: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) of audio
    
    Args:
        audio_data: Input audio array
    
    Returns:
        RMS value
    """
    if len(audio_data) == 0:
        return 0.0
    
    if audio_data.dtype != np.float32:
        audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    else:
        audio_float = audio_data
    
    return np.sqrt(np.mean(audio_float ** 2))


def calculate_loudness(audio_data: np.ndarray, sample_rate: int) -> float:
    """
    Calculate loudness in LUFS
    
    Args:
        audio_data: Input audio array
        sample_rate: Audio sample rate
    
    Returns:
        Loudness in LUFS
    """
    if len(audio_data) == 0:
        return -70.0  # Very quiet
    
    if PYLOUDNORM_AVAILABLE:
        try:
            # Ensure audio is float
            if audio_data.dtype != np.float32:
                audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
            else:
                audio_float = audio_data
            
            # Create meter
            meter = pyln.Meter(sample_rate)
            
            # Measure loudness
            loudness = meter.integrated_loudness(audio_float)
            
            return loudness
            
        except Exception as e:
            print(f"Loudness measurement failed: {e}")
    
    # Fallback: simple RMS to dB conversion
    rms = calculate_rms(audio_data)
    if rms > 0:
        return 20 * np.log10(rms)
    else:
        return -70.0


def convert_to_wav_bytes(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: type = np.int16
) -> bytes:
    """
    Convert audio array to WAV bytes
    
    Args:
        audio_data: Input audio array
        sample_rate: Audio sample rate
        channels: Number of channels
        dtype: Output data type
    
    Returns:
        WAV file bytes
    """
    # Ensure correct shape
    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(-1, channels)
    
    # Convert to target dtype
    if dtype == np.int16:
        if audio_data.dtype != np.int16:
            if np.issubdtype(audio_data.dtype, np.floating):
                audio_converted = (audio_data * 32767).astype(np.int16)
            else:
                audio_converted = audio_data.astype(np.int16)
        else:
            audio_converted = audio_data
    else:
        audio_converted = audio_data.astype(dtype)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(dtype().itemsize)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_converted.tobytes())
    
    return buffer.getvalue()


def save_audio_file(
    audio_data: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int = 16000,
    format: str = 'wav'
):
    """
    Save audio data to file
    
    Args:
        audio_data: Audio data array
        file_path: Output file path
        sample_rate: Audio sample rate
        format: Audio format ('wav', 'mp3', 'flac')
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'wav':
        if SOUNDFILE_AVAILABLE:
            try:
                sf.write(file_path, audio_data, sample_rate)
                return
            except Exception as e:
                print(f"Soundfile write failed: {e}")
        
        # Fallback using wave module
        try:
            with wave.open(str(file_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert to int16
                if audio_data.dtype != np.int16:
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data
                
                wav_file.writeframes(audio_int16.tobytes())
            return
        except Exception as e:
            raise Exception(f"Failed to save WAV file: {e}")
    
    elif format.lower() == 'mp3':
        if PYDUB_AVAILABLE and SOUNDFILE_AVAILABLE:
            try:
                # Save as WAV first, then convert to MP3
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, audio_data, sample_rate)
                    
                    # Convert to MP3
                    audio = AudioSegment.from_wav(tmp.name)
                    audio.export(file_path, format='mp3', bitrate='128k')
                    
                    os.unlink(tmp.name)
                return
            except Exception as e:
                print(f"MP3 conversion failed: {e}")
        
        raise ImportError("MP3 format requires pydub and soundfile libraries")
    
    elif format.lower() == 'flac':
        if SOUNDFILE_AVAILABLE:
            try:
                sf.write(file_path, audio_data, sample_rate, format='FLAC')
                return
            except Exception as e:
                raise Exception(f"Failed to save FLAC file: {e}")
        else:
            raise ImportError("FLAC format requires soundfile library")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_audio_file(
    file_path: Union[str, Path],
    target_sample_rate: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load audio file
    
    Args:
        file_path: Audio file path
        target_sample_rate: Target sample rate (resamples if different)
        mono: Convert to mono if stereo
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    # Try soundfile first
    if SOUNDFILE_AVAILABLE:
        try:
            audio_data, sample_rate = sf.read(file_path)
            
            # Convert to mono if needed
            if mono and len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Resample if needed
            if target_sample_rate and target_sample_rate != sample_rate:
                audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)
                sample_rate = target_sample_rate
            
            return audio_data, sample_rate
        except Exception as e:
            print(f"Soundfile load failed: {e}")
    
    # Fallback using wave module for WAV files
    if file_path.suffix.lower() == '.wav':
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                
                # Convert bytes to numpy array
                if wav_file.getsampwidth() == 2:  # 16-bit
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                elif wav_file.getsampwidth() == 1:  # 8-bit
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                    audio_data = audio_data.astype(np.float32) / 128.0 - 1.0
                else:
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                
                # Handle stereo
                if wav_file.getnchannels() > 1:
                    audio_data = audio_data.reshape(-1, wav_file.getnchannels())
                    if mono:
                        audio_data = np.mean(audio_data, axis=1)
                
                # Convert to float32 for consistency
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Resample if needed
                if target_sample_rate and target_sample_rate != sample_rate:
                    audio_data = resample_audio(audio_data, sample_rate, target_sample_rate)
                    sample_rate = target_sample_rate
                
                return audio_data, sample_rate
        except Exception as e:
            raise Exception(f"Failed to load WAV file: {e}")
    
    # Try pydub for other formats
    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_file(file_path)
            
            if mono and audio.channels > 1:
                audio = audio.set_channels(1)
            
            if target_sample_rate:
                audio = audio.set_frame_rate(target_sample_rate)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            
            if audio.sample_width == 2:
                samples = samples.astype(np.int16)
            elif audio.sample_width == 4:
                samples = samples.astype(np.int32)
            
            # Convert to float32
            samples = samples.astype(np.float32) / 32768.0
            
            # Reshape if stereo
            if audio.channels > 1:
                samples = samples.reshape(-1, audio.channels)
                if mono:
                    samples = np.mean(samples, axis=1)
            
            return samples, audio.frame_rate
        except Exception as e:
            print(f"Pydub load failed: {e}")
    
    raise ImportError(f"No suitable library found to load {file_path}. Install soundfile or pydub.")


def trim_silence(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold: float = 0.01,
    min_silence_duration: float = 0.1
) -> np.ndarray:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio_data: Input audio array
        sample_rate: Audio sample rate
        threshold: Silence threshold
        min_silence_duration: Minimum silence duration to consider as silence
    
    Returns:
        Trimmed audio array
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Detect silence
    silence_mask = detect_silence(
        audio_data,
        threshold=threshold,
        min_silence_duration=min_silence_duration,
        sample_rate=sample_rate
    )
    
    # Find first non-silent sample
    start_idx = 0
    for i in range(len(silence_mask)):
        if not silence_mask[i]:
            start_idx = i
            break
    
    # Find last non-silent sample
    end_idx = len(silence_mask)
    for i in range(len(silence_mask) - 1, -1, -1):
        if not silence_mask[i]:
            end_idx = i + 1
            break
    
    return audio_data[start_idx:end_idx]


def apply_fade(
    audio_data: np.ndarray,
    sample_rate: int,
    fade_in: float = 0.01,
    fade_out: float = 0.01
) -> np.ndarray:
    """
    Apply fade in and fade out to audio
    
    Args:
        audio_data: Input audio array
        sample_rate: Audio sample rate
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds
    
    Returns:
        Audio with fade applied
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Convert to float for processing
    if audio_data.dtype != np.float32:
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_float = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        else:
            audio_float = audio_data.astype(np.float32)
    else:
        audio_float = audio_data.copy()
    
    # Calculate fade samples
    fade_in_samples = int(sample_rate * fade_in)
    fade_out_samples = int(sample_rate * fade_out)
    
    # Apply fade in
    if fade_in_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        if fade_in_samples > len(audio_float):
            fade_in_samples = len(audio_float)
        audio_float[:fade_in_samples] *= fade_in_curve[:fade_in_samples]
    
    # Apply fade out
    if fade_out_samples > 0:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        start_out = len(audio_float) - fade_out_samples
        if start_out < 0:
            start_out = 0
            fade_out_samples = len(audio_float)
        
        audio_float[start_out:] *= fade_out_curve[:len(audio_float) - start_out]
    
    # Convert back to original dtype
    if audio_data.dtype != np.float32:
        if np.issubdtype(audio_data.dtype, np.integer):
            return (audio_float * np.iinfo(audio_data.dtype).max).astype(audio_data.dtype)
        else:
            return audio_float.astype(audio_data.dtype)
    
    return audio_float


# Simple version without optional dependencies
def simple_resample(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Simple resampling without external dependencies"""
    if original_rate == target_rate:
        return audio_data
    
    ratio = target_rate / original_rate
    new_length = int(len(audio_data) * ratio)
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(audio_data))
    x_new = np.linspace(0, 1, new_length)
    
    return np.interp(x_new, x_old, audio_data)


def check_dependencies():
    """Check which audio dependencies are available"""
    dependencies = {
        'resampy': RESAMPY_AVAILABLE,
        'noisereduce': NOISEREDUCE_AVAILABLE,
        'scipy': SCIPY_AVAILABLE,
        'pyloudnorm': PYLOUDNORM_AVAILABLE,
        'soundfile': SOUNDFILE_AVAILABLE,
        'pydub': PYDUB_AVAILABLE
    }
    
    return dependencies