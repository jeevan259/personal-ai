"""
Logging configuration for the Personal Voice Assistant
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime
import colorlog

class CustomJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)
            
        if hasattr(record, "extra"):
            log_object.update(record.extra)
            
        return json.dumps(log_object, ensure_ascii=False)


class ColoredConsoleFormatter(colorlog.ColoredFormatter):
    """Colored formatter for console output"""
    
    def __init__(self):
        super().__init__(
            fmt="%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True
) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        enable_file_logging: Whether to log to files
        enable_console_logging: Whether to log to console
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    logging.root.handlers.clear()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Create handlers
    handlers = []
    
    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(ColoredConsoleFormatter())
        handlers.append(console_handler)
    
    # File handler
    if enable_file_logging and log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # General log file
        general_log = log_dir / "assistant.log"
        file_handler = logging.FileHandler(general_log, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(CustomJsonFormatter())
        handlers.append(file_handler)
        
        # Error log file
        error_log = log_dir / "error.log"
        error_handler = logging.FileHandler(error_log, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(CustomJsonFormatter())
        handlers.append(error_handler)
        
        # Audio processing log
        audio_log = log_dir / "audio_processing.log"
        audio_handler = logging.FileHandler(audio_log, encoding='utf-8')
        audio_handler.setLevel(logging.DEBUG)
        audio_handler.addFilter(lambda record: record.name.startswith('audio.'))
        audio_handler.setFormatter(CustomJsonFormatter())
        handlers.append(audio_handler)
        
        # Conversation log
        conversation_log = log_dir / "conversations.log"
        conversation_handler = logging.FileHandler(conversation_log, encoding='utf-8')
        conversation_handler.setLevel(logging.INFO)
        conversation_handler.addFilter(lambda record: record.name.startswith('conversation.'))
        conversation_handler.setFormatter(CustomJsonFormatter())
        handlers.append(conversation_handler)
    
    # Add all handlers to root logger
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set up specific loggers
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Log initialization
    logging.info(f"Logging initialized at level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Convenience logger for the module
logger = get_logger(__name__)