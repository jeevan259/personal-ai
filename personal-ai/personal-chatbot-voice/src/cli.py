"""
Command Line Interface for the Voice Chatbot
"""

import argparse
from pathlib import Path
from typing import Optional


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Personal Voice Chatbot - A customizable voice assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive
  %(prog)s --config ./config/settings.yaml --log-level DEBUG
  %(prog)s --web --port 8080
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive voice mode (default)"
    )
    mode_group.add_argument(
        "--web",
        "-w",
        action="store_true",
        help="Start web interface"
    )
    mode_group.add_argument(
        "--api",
        "-a",
        action="store_true",
        help="Start API server only"
    )
    mode_group.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Run tests"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--persona",
        "-p",
        type=str,
        default="default",
        help="Persona to use (default, professional, friendly)"
    )
    
    # Audio settings
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Audio input device index"
    )
    parser.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="Audio output device index"
    )
    parser.add_argument(
        "--no-wake-word",
        action="store_true",
        help="Disable wake word detection"
    )
    
    # Web/API settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for web/API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for web/API server"
    )
    parser.add_argument(
        "--websocket-port",
        type=int,
        default=8765,
        help="Port for WebSocket server"
    )
    
    # Development options
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling"
    )
    
    # Skills management
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List available skills"
    )
    parser.add_argument(
        "--enable-skill",
        type=str,
        action="append",
        help="Enable specific skill(s)"
    )
    parser.add_argument(
        "--disable-skill",
        type=str,
        action="append",
        help="Disable specific skill(s)"
    )
    
    # Voice cloning
    parser.add_argument(
        "--clone-voice",
        type=Path,
        help="Path to audio file for voice cloning"
    )
    parser.add_argument(
        "--train-wake-word",
        type=Path,
        help="Path to audio files for training custom wake word"
    )
    
    return parser.parse_args()


def validate_args(args) -> bool:
    """Validate command line arguments"""
    
    if args.config and not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return False
    
    if args.clone_voice and not args.clone_voice.exists():
        print(f"Error: Audio file for voice cloning not found: {args.clone_voice}")
        return False
    
    if args.train_wake_word and not args.train_wake_word.exists():
        print(f"Error: Training data directory not found: {args.train_wake_word}")
        return False
    
    return True


def print_audio_devices():
    """Print available audio devices"""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print("\nAvailable Audio Devices:")
        print("=" * 50)
        
        for i in range(p.get_device_count()):
            dev_info = p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0 or dev_info['maxOutputChannels'] > 0:
                direction = "Input" if dev_info['maxInputChannels'] > 0 else "Output"
                if dev_info['maxInputChannels'] > 0 and dev_info['maxOutputChannels'] > 0:
                    direction = "Input/Output"
                
                print(f"{i}: {dev_info['name']} ({direction})")
                print(f"    Channels: In={dev_info['maxInputChannels']}, Out={dev_info['maxOutputChannels']}")
                print(f"    Sample Rate: {dev_info['defaultSampleRate']} Hz")
                print()
        
        p.terminate()
        
    except ImportError:
        print("PyAudio not installed. Cannot list audio devices.")
    except Exception as e:
        print(f"Error listing audio devices: {e}")