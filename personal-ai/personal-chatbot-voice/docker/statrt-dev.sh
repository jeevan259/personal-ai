#!/bin/bash
# Development startup script with hot reload

set -e

echo "Starting Personal Voice Assistant in development mode..."

# Activate Python virtual environment if exists
if [ -f "/app/venv/bin/activate" ]; then
    source /app/venv/bin/activate
fi

# Install/update dependencies in development mode
if [ ! -f "/app/.dependencies-installed" ]; then
    echo "Installing development dependencies..."
    pip install -e .[dev]
    touch /app/.dependencies-installed
fi

# Wait for dependencies (if using external services)
if [ "$WAIT_FOR_SERVICES" = "true" ]; then
    echo "Waiting for dependent services..."
    /app/scripts/wait-for-it.sh chroma-db-dev:8000 --timeout=30
    /app/scripts/wait-for-it.sh ollama-dev:11434 --timeout=30
    /app/scripts/wait-for-it.sh redis-dev:6379 --timeout=30
fi

# Run database migrations (if using database)
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m scripts.run_migrations
fi

# Start with debugger or hot reload
if [ "$DEBUG_MODE" = "true" ]; then
    echo "Starting with debugger on port 5678..."
    python -m debugpy --listen 0.0.0.0:5678 -m src.main
elif [ "$HOT_RELOAD" = "true" ]; then
    echo "Starting with hot reload..."
    watchmedo auto-restart \
        --directory=/app/src \
        --patterns="*.py" \
        --ignore-patterns="*/__pycache__/*" \
        --recursive \
        -- python -m src.main
else
    echo "Starting normally..."
    python -m src.main
fi