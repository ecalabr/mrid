"""
Logging configuration for the MRID application.

This module provides a centralized logging system with:
- Console (terminal) output
- File logging to an appendable log file in the output directory
- Configurable log levels via config.py
- Buffering of log messages before output directory is specified
"""

import logging
import os
import sys
from typing import Optional, List

# Log format strings
CONSOLE_FORMAT = "%(levelname)s: %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log filename
LOG_FILENAME = "mrid.log"


class BufferingHandler(logging.Handler):
    """
    A handler that buffers log records until a file handler is available.
    Once the file handler is set, it flushes all buffered records to the file.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize the buffering handler.

        Args:
            capacity: Maximum number of log records to buffer
        """
        super().__init__()
        self.capacity = capacity
        self.buffer: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Store the log record in the buffer.

        Args:
            record: The log record to buffer
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(record)

    def flush_to_handler(self, handler: logging.Handler) -> None:
        """
        Flush all buffered records to the specified handler.

        Args:
            handler: The handler to flush records to (typically a FileHandler)
        """
        for record in self.buffer:
            handler.handle(record)
        self.buffer.clear()

    def get_buffered_count(self) -> int:
        """Return the number of buffered records."""
        return len(self.buffer)


# Module-level logger instance
_logger: Optional[logging.Logger] = None
_file_handler: Optional[logging.FileHandler] = None
_console_handler: Optional[logging.StreamHandler] = None
_buffering_handler: Optional[BufferingHandler] = None

def get_log_level_from_string(level_str: str) -> int:
    """
    Convert a log level string to a logging level constant.

    Args:
        level_str: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        int: Corresponding logging level constant
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'WARN': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    return level_map.get(level_str.upper(), logging.INFO)


def setup_logging(output_dir: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging with console and file handlers.

    Args:
        output_dir: Directory for the log file. If None, file logging is buffered.
        level: Log level string. If None, uses LOG_LEVEL from config.

    Returns:
        logging.Logger: Configured logger instance
    """
    global _logger, _file_handler, _console_handler, _buffering_handler

    # Import config here to avoid circular imports
    try:
        from . import config
        if level is None:
            level = getattr(config, 'LOG_LEVEL', 'INFO')
    except ImportError:
        if level is None:
            level = 'INFO'

    log_level = get_log_level_from_string(level)

    # Create or get the logger
    if _logger is None:
        _logger = logging.getLogger('mrid')
        _logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow handlers to filter
        _logger.propagate = False  # Don't propagate to root logger

    # Remove existing handlers to avoid duplicates on re-init
    if _console_handler is not None:
        _logger.removeHandler(_console_handler)
        _console_handler = None
    if _file_handler is not None:
        _logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None

    # Console handler
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(CONSOLE_FORMAT)
    _console_handler.setFormatter(console_formatter)
    _logger.addHandler(_console_handler)

    # File handler (if output_dir is provided)
    if output_dir and os.path.isdir(output_dir):
        try:
            log_path = os.path.join(output_dir, LOG_FILENAME)
            _file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            _file_handler.setLevel(logging.DEBUG)  # File handler always captures DEBUG and above
            file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
            _file_handler.setFormatter(file_formatter)
            _logger.addHandler(_file_handler)

            # Flush buffered records to file if buffer exists
            if _buffering_handler is not None:
                _buffering_handler.flush_to_handler(_file_handler)
                _logger.removeHandler(_buffering_handler)
                _buffering_handler = None

        except Exception as e:
            _logger.warning(f"Could not create log file at {output_dir}: {e}")
    else:
        # No output directory yet - set up buffering handler if not already present
        if _buffering_handler is None:
            _buffering_handler = BufferingHandler()
            _buffering_handler.setLevel(logging.DEBUG)  # Buffer captures DEBUG and above
            file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
            _buffering_handler.setFormatter(file_formatter)
            _logger.addHandler(_buffering_handler)

    return _logger


def get_logger() -> logging.Logger:
    """
    Get the MRID logger instance.

    If the logger hasn't been set up yet, creates a basic console-only logger.

    Returns:
        logging.Logger: The MRID logger
    """
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


def update_log_file_location(output_dir: str) -> None:
    """
    Update the log file location when the output directory changes.
    Flushes any buffered log records to the new file.

    Args:
        output_dir: New output directory for the log file
    """
    global _logger, _file_handler, _buffering_handler

    if _logger is None:
        setup_logging(output_dir)
        return

    # Remove old file handler
    if _file_handler is not None:
        _logger.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None

    # Add new file handler
    if output_dir and os.path.isdir(output_dir):
        try:
            from . import config
            level = getattr(config, 'LOG_LEVEL', 'INFO')
            log_level = get_log_level_from_string(level)

            log_path = os.path.join(output_dir, LOG_FILENAME)
            _file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            _file_handler.setLevel(logging.DEBUG)  # File handler always captures DEBUG and above
            file_formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
            _file_handler.setFormatter(file_formatter)
            _logger.addHandler(_file_handler)

            # Flush buffered records to file if buffer exists
            if _buffering_handler is not None:
                _buffering_handler.flush_to_handler(_file_handler)
                _logger.removeHandler(_buffering_handler)
                _buffering_handler = None
                _logger.debug(f"Flushed buffered log records to {log_path}")

        except Exception as e:
            _logger.warning(f"Could not create log file at {output_dir}: {e}")


def set_log_level(level: str) -> None:
    """
    Change the log level for all handlers.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    global _logger, _file_handler, _console_handler

    log_level = get_log_level_from_string(level)

    if _console_handler is not None:
        _console_handler.setLevel(log_level)
    if _file_handler is not None:
        _file_handler.setLevel(log_level)


# Convenience functions for common log operations
def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message."""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log an info message."""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message."""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log an error message."""
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message."""
    get_logger().critical(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs) -> None:
    """Log an exception with traceback."""
    get_logger().exception(msg, *args, **kwargs)
