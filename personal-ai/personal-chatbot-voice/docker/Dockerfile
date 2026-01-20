# Multi-stage build for production
# Stage 1: Builder
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    libffi-dev \
    libssl-dev \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Install runtime dependencies (minimal)
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    libavcodec-extra \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Create necessary directories
RUN mkdir -p /app/data/{audio_samples,models,logs,knowledge_base,memory_db,voice_profiles} && \
    mkdir -p /app/data/audio_samples/{wake_word,voice_cloning,test_audio,conversations} && \
    mkdir -p /app/data/knowledge_base/{documents,notes,emails,structured} && \
    mkdir -p /app/data/memory_db/{chroma_db,sqlite} && \
    mkdir -p /app/data/voice_profiles/{user_voice,assistant_voice} && \
    mkdir -p /app/models/{whisper,vosk,tts,wake_word}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Expose ports
EXPOSE 8000  # API server
EXPOSE 8765  # WebSocket server

# Default command
CMD ["python", "-m", "src.main"]