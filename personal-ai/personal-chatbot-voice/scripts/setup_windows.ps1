# Windows Setup Script for Personal Voice Assistant

Write-Host "Setting up Personal Voice Assistant on Windows..." -ForegroundColor Green

# Check Python version
$pythonVersion = (python --version 2>&1).ToString()
if ($pythonVersion -match "Python 3\.(1[1-9]|[2-9][0-9])\.[0-9]+") {
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "Error: Python 3.11 or higher required" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
.\venv\Scripts\Activate

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install PyAudio first (special Windows installation)
Write-Host "Installing PyAudio (Windows may need special handling)..." -ForegroundColor Yellow
try {
    # Try normal install first
    pip install PyAudio
} catch {
    Write-Host "PyAudio installation failed, trying alternative..." -ForegroundColor Yellow
    # Try using pipwin
    pip install pipwin
    pipwin install pyaudio
}

# Install other dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install development dependencies if requested
if ($args[0] -eq "--dev") {
    Write-Host "Installing development dependencies..." -ForegroundColor Yellow
    pip install -r requirements-dev.txt
}

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
$directories = @(
    "data",
    "data\audio_samples",
    "data\audio_samples\wake_word", 
    "data\audio_samples\voice_cloning",
    "data\audio_samples\test_audio",
    "data\audio_samples\conversations",
    "data\knowledge_base",
    "data\knowledge_base\documents",
    "data\knowledge_base\notes",
    "data\knowledge_base\emails",
    "data\knowledge_base\structured",
    "data\memory_db",
    "data\memory_db\chroma_db",
    "data\memory_db\sqlite",
    "data\voice_profiles",
    "data\voice_profiles\user_voice",
    "data\voice_profiles\assistant_voice",
    "data\logs",
    "models",
    "models\whisper",
    "models\vosk",
    "models\tts",
    "models\wake_word"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir"
    }
}

# Set up configuration
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "Please edit .env file with your configuration" -ForegroundColor Cyan
    } else {
        Write-Host "Creating basic .env file..." -ForegroundColor Yellow
        @"
# Windows Environment Variables
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_DIR=./data
MODELS_DIR=./models

# Audio Configuration
AUDIO_BACKEND=pyaudio
SAMPLE_RATE=16000
CHUNK_SIZE=1024

# Wake Word
WAKE_WORD_ENABLED=true
WAKE_WORD_PHRASE="hey assistant"

# LLM Configuration  
LLM_PROVIDER=local
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# TTS Configuration
TTS_PROVIDER=pyttsx3
"@ | Out-File ".env" -Encoding UTF8
    }
}

# Download models if requested
if ($args[0] -eq "--download-models") {
    Write-Host "Downloading Whisper models..." -ForegroundColor Yellow
    python -c "import whisper; whisper.load_model('tiny'); whisper.load_model('base')"
    
    Write-Host "Note: Other models will be downloaded on first use" -ForegroundColor Cyan
}

Write-Host "" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "To activate the virtual environment:" -ForegroundColor Cyan
Write-Host "  .\venv\Scripts\Activate" -ForegroundColor White
Write-Host "" -ForegroundColor Green
Write-Host "To run the voice assistant:" -ForegroundColor Cyan
Write-Host "  python -m src.main" -ForegroundColor White
Write-Host "" -ForegroundColor Green
Write-Host "To run CLI mode:" -ForegroundColor Cyan
Write-Host "  python -m src.cli chat" -ForegroundColor White