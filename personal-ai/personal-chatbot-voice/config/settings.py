import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import yaml


class AudioConfig(BaseSettings):
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    dtype: str = "int16"
    silence_threshold: int = 500
    vad_aggressiveness: int = 3
    max_record_seconds: int = 30
    silence_duration: float = 1.0
    noise_reduction_level: int = 1


class LLMConfig(BaseSettings):
    model: str = "gpt-4-turbo"
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    streaming: bool = True
    timeout: int = 30


class VoiceConfig(BaseSettings):
    default_voice_id: str = "rachel"
    voice_cloning_enabled: bool = False
    xtts_model_path: Path = Path("./models/tts/xtts_v2")
    speech_rate: int = 150
    pitch: int = 50
    volume: float = 1.0
    language: str = "en"


class WakeWordConfig(BaseSettings):
    enabled: bool = True
    sensitivity: float = 0.5
    porcupine_access_key: Optional[str] = None
    custom_wake_word_path: Optional[Path] = None
    timeout: int = 5


class Settings(BaseSettings):
    # API Keys
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    azure_speech_key: Optional[str] = None
    azure_speech_region: Optional[str] = None

    # Paths
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = base_dir / "data"
    models_dir: Path = base_dir / "models"
    logs_dir: Path = base_dir / "data" / "logs"

    # Configurations
    audio: AudioConfig = AudioConfig()
    llm: LLMConfig = LLMConfig()
    voice: VoiceConfig = VoiceConfig()
    wake_word: WakeWordConfig = WakeWordConfig()

    # Privacy
    auto_delete_audio_days: int = 7
    encrypt_storage: bool = True

    # Performance
    latency_target_ms: int = 300
    enable_streaming: bool = True
    debug_mode: bool = False

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

    @classmethod
    def from_yaml(cls, path: Path):
        """Load settings from YAML file"""
        with open(path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


# Global settings instance
settings = Settings()


def load_persona_config(persona_name: str) -> dict:
    """Load persona-specific configuration"""
    persona_path = settings.base_dir / "config" / "personas" / f"{persona_name}.yaml"
    if persona_path.exists():
        with open(persona_path, 'r') as f:
            return yaml.safe_load(f)
    return {}