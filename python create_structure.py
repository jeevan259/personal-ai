from pathlib import Path

BASE_DIR = Path("personal-ai") / "personal-chatbot-voice"

# Helper functions
def create_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def create_file(path: Path):
    if not path.exists():
        path.touch()

# Directory structure
dirs = [
    "config/personas",
    "src/audio/capture",
    "src/audio/processing",
    "src/audio/wake_word",
    "src/audio/output",
    "src/speech/stt",
    "src/speech/tts",
    "src/speech/voice_cloning",
    "src/core/prompt_templates",
    "src/llm",
    "src/memory",
    "src/skills/skills",
    "src/nlu",
    "src/interface",
    "src/utils",
    "data/audio_samples/wake_word",
    "data/audio_samples/voice_cloning",
    "data/audio_samples/test_audio",
    "data/knowledge_base/documents",
    "data/knowledge_base/notes",
    "data/knowledge_base/embeddings",
    "data/memory_db/chroma_db",
    "data/memory_db/sqlite",
    "data/voice_profiles/user_voice",
    "data/voice_profiles/assistant_voice",
    "data/logs/audio_logs",
    "data/logs/conversation_logs",
    "data/logs/performance_logs",
    "models/whisper",
    "models/vosk",
    "models/tts",
    "models/wake_word",
    "tests/unit",
    "tests/integration",
    "tests/performance",
    "tests/fixtures/audio_samples",
    "scripts",
    "docker",
    "docs",
]

files = [
    ".env.example",
    ".gitignore",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "docker-compose.yml",

    "config/__init__.py",
    "config/settings.py",
    "config/audio_config.yaml",
    "config/llm_config.yaml",
    "config/voice_config.yaml",
    "config/wake_word_config.yaml",
    "config/personas/default_voice.yaml",
    "config/personas/professional_voice.yaml",
    "config/personas/friendly_voice.yaml",

    "src/__init__.py",
    "src/main.py",
    "src/cli.py",

    "src/audio/__init__.py",
    "src/audio/capture/__init__.py",
    "src/audio/capture/microphone.py",
    "src/audio/capture/file_input.py",
    "src/audio/capture/stream_input.py",

    "src/audio/processing/__init__.py",
    "src/audio/processing/vad.py",
    "src/audio/processing/noise_reduction.py",
    "src/audio/processing/preprocessor.py",
    "src/audio/processing/chunker.py",

    "src/audio/wake_word/__init__.py",
    "src/audio/wake_word/porcupine_wake.py",
    "src/audio/wake_word/custom_wake.py",
    "src/audio/wake_word/wake_manager.py",

    "src/audio/output/__init__.py",
    "src/audio/output/player.py",
    "src/audio/output/stream_output.py",

    "src/speech/__init__.py",
    "src/speech/stt/__init__.py",
    "src/speech/stt/whisper_stt.py",
    "src/speech/stt/faster_whisper.py",
    "src/speech/stt/vosk_stt.py",
    "src/speech/stt/stt_manager.py",

    "src/speech/tts/__init__.py",
    "src/speech/tts/elevenlabs_tts.py",
    "src/speech/tts/openai_tts.py",
    "src/speech/tts/pyttsx3_local.py",
    "src/speech/tts/edge_tts.py",
    "src/speech/tts/tts_manager.py",

    "src/speech/voice_cloning/__init__.py",
    "src/speech/voice_cloning/xtts.py",
    "src/speech/voice_cloning/voice_manager.py",

    "src/core/__init__.py",
    "src/core/voice_engine.py",
    "src/core/conversation_manager.py",
    "src/core/audio_context.py",
    "src/core/interruption_handler.py",
    "src/core/latency_optimizer.py",
    "src/core/prompt_templates/__init__.py",
    "src/core/prompt_templates/voice_system.jinja2",
    "src/core/prompt_templates/short_response.jinja2",
    "src/core/prompt_templates/conversation.jinja2",

    "src/llm/__init__.py",
    "src/llm/voice_optimized_llm.py",
    "src/llm/response_shortener.py",
    "src/llm/streaming_llm.py",
    "src/llm/prosody_hints.py",

    "src/memory/__init__.py",
    "src/memory/audio_memory.py",
    "src/memory/transcript_store.py",
    "src/memory/voice_patterns.py",

    "src/skills/__init__.py",
    "src/skills/base_skill.py",
    "src/skills/skill_registry.py",
    "src/skills/skills/__init__.py",
    "src/skills/skills/weather.py",
    "src/skills/skills/timer.py",
    "src/skills/skills/reminder.py",
    "src/skills/skills/knowledge_query.py",
    "src/skills/skills/calendar.py",
    "src/skills/skills/music.py",

    "src/nlu/__init__.py",
    "src/nlu/intent_classifier.py",
    "src/nlu/entity_extractor.py",
    "src/nlu/voice_command_parser.py",
    "src/nlu/context_understanding.py",

    "src/interface/__init__.py",
    "src/interface/voice_interface.py",
    "src/interface/web_interface.py",
    "src/interface/api_server.py",
    "src/interface/websocket_server.py",

    "src/utils/__init__.py",
    "src/utils/audio_utils.py",
    "src/utils/logging_config.py",
    "src/utils/performance_monitor.py",
    "src/utils/privacy.py",
    "src/utils/config_loader.py",

    "tests/__init__.py",
    "tests/unit/test_audio_processing.py",
    "tests/unit/test_stt.py",
    "tests/unit/test_tts.py",
    "tests/unit/test_llm_integration.py",
    "tests/integration/test_voice_pipeline.py",
    "tests/integration/test_skills.py",
    "tests/performance/test_latency.py",
    "tests/performance/test_accuracy.py",

    "scripts/setup_environment.sh",
    "scripts/train_wake_word.py",
    "scripts/clone_voice.py",
    "scripts/import_knowledge.py",
    "scripts/benchmark.py",

    "docker/Dockerfile",
    "docker/docker-compose.dev.yml",
    "docker/nginx.conf",

    "docs/architecture.md",
    "docs/setup_guide.md",
    "docs/api_reference.md",
    "docs/voice_training.md",
    "docs/troubleshooting.md",
]

# Create structure
for d in dirs:
    create_dir(BASE_DIR / d)

for f in files:
    create_file(BASE_DIR / f)

print("✅ Project structure created successfully.")
