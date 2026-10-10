import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Environment-configurable log settings
LOG_LEVEL_NAME = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

# Global initialization flag
_is_logging_configured = False


def setup_root_logger() -> None:
    """
    Sets up the root logger with both a RotatingFileHandler and a StreamHandler.
    Safe to call multiple times (idempotent).
    """
    global _is_logging_configured
    if _is_logging_configured:
        return

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Avoid duplicate handlers if already present
    if not root_logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Rotating File Handler
        file_handler = RotatingFileHandler(
            filename=DEFAULT_LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8"
        )
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # 2. Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _is_logging_configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger instance with the specified name.
    Initializes root logging if not yet initialized.
    """
    setup_root_logger()
    return logging.getLogger(name or "app")