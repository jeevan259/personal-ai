#!/bin/bash
# Setup environment for Personal Voice Assistant

set -e

echo "Setting up Personal Voice Assistant environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.11" ]]; then
    echo "Error: Python 3.11 or higher required"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$1" = "--dev" ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data/audio_samples/{wake_word,voice_cloning,test_audio,conversations}
mkdir -p data/knowledge_base/{documents,notes,emails,structured}
mkdir -p data/memory_db/{chroma_db,sqlite}
mkdir -p data/voice_profiles/{user_voice,assistant_voice}
mkdir -p data/logs
mkdir -p models/{whisper,vosk,tts,wake_word}

# Set up configuration
echo "Setting up configuration..."
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Download models (optional)
if [ "$2" = "--download-models" ]; then
    echo "Downloading models..."
    
    # Download Whisper models
    echo "Downloading Whisper models..."
    python -c "import whisper; whisper.load_model('tiny'); whisper.load_model('base')"
    
    # Note: Other models would be downloaded on first use
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the voice assistant:"
echo "  python -m src.main"
echo ""
echo "To run CLI mode:"
echo "  python -m src.cli chat"