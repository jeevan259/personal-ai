"""
Personal Voice Chatbot Package
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Use lazy imports to avoid circular import issues
import importlib
import sys

def _lazy_import(module_name, attribute_name=None):
    """Lazily import a module or attribute"""
    try:
        module = importlib.import_module(f".{module_name}", package="src")
        if attribute_name:
            return getattr(module, attribute_name)
        return module
    except ImportError as e:
        print(f"Note: Could not import {module_name}: {e}")
        return None

# Lazy imports for main components
_main_module = _lazy_import("main", "main")
if _main_module:
    main = _main_module
else:
    # Fallback function
    def main():
        print("Main module not available")
        return 1

_cli_module = _lazy_import("cli")
if _cli_module:
    parse_args = getattr(_cli_module, 'parse_args', None)
    validate_args = getattr(_cli_module, 'validate_args', None)
    print_audio_devices = getattr(_cli_module, 'print_audio_devices', None)
else:
    # Fallback functions
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser(description="Voice Chatbot")
        return parser.parse_args()
    
    def validate_args(args):
        return True
    
    def print_audio_devices():
        print("Audio device listing not available")

# Define exports
__all__ = [
    'main',
    'parse_args',
    'validate_args',
    'print_audio_devices',
    '__version__',
    '__author__'
]