import logging
import logging.handlers
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """
    Set up logging configuration to write to both console and file.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"deep_research_agent_{timestamp}.log"
    log_filepath = log_path / log_filename

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler (simpler format)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)

    # File handler (detailed format)
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(detailed_formatter)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Console and file: {log_filepath}")
    logger.info(f"Log level: {log_level.upper()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)
