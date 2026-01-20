"""
Logging configuration for the voice chatbot
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys
import colorlog


def setup_logging(
    name: str = __name__,
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Create formatters
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Add null handler to prevent "No handlers" warning
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    return logger


def get_audio_logger() -> logging.Logger:
    """Get specialized logger for audio processing"""
    logger = logging.getLogger('audio')
    
    if not logger.handlers:
        # Create logs directory
        logs_dir = Path("data/logs/audio_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio log file
        log_file = logs_dir / "audio_processing.log"
        
        # Setup handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def get_performance_logger() -> logging.Logger:
    """Get specialized logger for performance metrics"""
    logger = logging.getLogger('performance')
    
    if not logger.handlers:
        # Create logs directory
        logs_dir = Path("data/logs/performance_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance log file
        log_file = logs_dir / "performance_metrics.log"
        
        # Setup handler
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=2 * 1024 * 1024,  # 2 MB
            backupCount=7  # Keep a week of logs
        )
        
        # CSV formatter for performance metrics
        class CSVFormatter(logging.Formatter):
            def format(self, record):
                # Create CSV line
                return f"{self.formatTime(record)}," \
                       f"{record.levelname}," \
                       f"{record.module}," \
                       f"{record.funcName}," \
                       f"{record.lineno}," \
                       f"{record.getMessage()}"
        
        formatter = CSVFormatter(
            fmt='%(asctime)s,%(levelname)s,%(module)s,%(funcName)s,%(lineno)d,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def get_conversation_logger() -> logging.Logger:
    """Get specialized logger for conversation logging"""
    logger = logging.getLogger('conversation')
    
    if not logger.handlers:
        # Create logs directory
        logs_dir = Path("data/logs/conversation_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Conversation log file with date
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = logs_dir / f"conversation_{date_str}.log"
        
        # Setup handler
        handler = logging.FileHandler(log_file, encoding='utf-8')
        
        # JSON formatter for structured logging
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                import json
                from datetime import datetime
                
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'level': record.levelname,
                    'session_id': getattr(record, 'session_id', 'unknown'),
                    'user_id': getattr(record, 'user_id', 'unknown'),
                    'message': record.getMessage(),
                    'data': getattr(record, 'data', {})
                }
                
                return json.dumps(log_entry, ensure_ascii=False)
        
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


def log_conversation(session_id: str, user_id: str, message: str, data: dict = None):
    """Log a conversation entry"""
    logger = get_conversation_logger()
    
    # Add custom attributes to log record
    import logging
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.session_id = session_id
        record.user_id = user_id
        record.data = data or {}
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    logger.info(message)
    
    # Restore original factory
    logging.setLogRecordFactory(old_factory)


def log_performance_metric(metric_name: str, value: float, unit: str = "ms", metadata: dict = None):
    """Log a performance metric"""
    logger = get_performance_logger()
    
    if metadata:
        message = f"{metric_name}={value}{unit} metadata={metadata}"
    else:
        message = f"{metric_name}={value}{unit}"
    
    logger.info(message)


def log_audio_event(event_type: str, details: str, audio_info: dict = None):
    """Log an audio processing event"""
    logger = get_audio_logger()
    
    if audio_info:
        details = f"{details} | {audio_info}"
    
    if event_type == 'error':
        logger.error(details)
    elif event_type == 'warning':
        logger.warning(details)
    else:
        logger.info(details)


def setup_global_logging(
    log_level: str = "INFO",
    log_dir: Path = Path("data/logs")
):
    """
    Setup global logging configuration for the entire application
    
    Args:
        log_level: Global log level
        log_dir: Directory for log files
    """
    
    # Create log directory
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Main application log
    main_log = log_dir / "application.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.handlers.RotatingFileHandler(
        main_log,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10
    )
    file_handler.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    
    root_logger.addHandler(console_handler)
    
    # Set specific loggers
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    
    return root_logger