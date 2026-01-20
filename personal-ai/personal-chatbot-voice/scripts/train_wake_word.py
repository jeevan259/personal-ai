#!/usr/bin/env python3
"""
Train custom wake word
"""

import argparse
import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import get_settings
from audio.wake_word.porcupine_wake import PorcupineWakeWordDetector
from utils.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)

async def train_wake_word(samples_dir: Path, wake_word: str):
    """Train custom wake word"""
    settings = get_settings()
    setup_logging(settings.log_level, settings.logs_dir)
    
    if not samples_dir.exists():
        logger.error(f"Samples directory not found: {samples_dir}")
        return False
        
    # Count audio files
    audio_files = list(samples_dir.glob("*.wav")) + list(samples_dir.glob("*.mp3"))
    if len(audio_files) < 10:
        logger.warning(f"Only {len(audio_files)} audio files found. At least 10 recommended.")
        
    logger.info(f"Training wake word '{wake_word}' with {len(audio_files)} samples")
    
    # Create detector
    detector = PorcupineWakeWordDetector()
    
    # Train (this is a mock implementation - real training requires Picovoice Console)
    output_dir = settings.models_dir / "wake_word"
    model_path = await detector.train_custom_wake_word(
        samples_dir=samples_dir,
        output_dir=output_dir,
        wake_word=wake_word
    )
    
    if model_path:
        logger.info(f"Wake word model saved to: {model_path}")
        print(f"\nTraining complete!")
        print(f"Model: {model_path}")
        print(f"\nTo use this model, update config/wake_word_config.yaml:")
        print(f"  custom:")
        print(f"    model_path: \"{model_path}\"")
        print(f"    enabled: true")
        return True
    else:
        logger.error("Failed to train wake word")
        return False

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train custom wake word")
    parser.add_argument(
        "samples_dir",
        type=Path,
        help="Directory containing audio samples of wake word"
    )
    parser.add_argument(
        "--wake-word",
        type=str,
        default="hey assistant",
        help="Wake word phrase (default: 'hey assistant')"
    )
    
    args = parser.parse_args()
    
    success = await train_wake_word(args.samples_dir, args.wake_word)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))