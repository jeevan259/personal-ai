"""
Audio test fixtures
"""

import numpy as np
from pathlib import Path

def generate_test_audio(duration_seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio signal"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    
    # Generate tone at 440Hz
    frequency = 440
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16


def generate_silence(duration_seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence"""
    samples = int(sample_rate * duration_seconds)
    return np.zeros(samples, dtype=np.int16)


def generate_noise(duration_seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate white noise"""
    samples = int(sample_rate * duration_seconds)
    noise = np.random.randn(samples).astype(np.float32) * 0.1
    return (noise * 32767).astype(np.int16)


# Common test audio files
TEST_AUDIO_FIXTURES = {
    "tone_1s": generate_test_audio(1.0),
    "silence_1s": generate_silence(1.0),
    "noise_1s": generate_noise(1.0),
    "tone_2s": generate_test_audio(2.0),
    "silence_2s": generate_silence(2.0),
}