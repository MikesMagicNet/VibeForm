# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  logging_config.py  –  Centralized structured logging for all modules      ║
# ║                                                                            ║
# ║  Provides tiered logging (DEBUG, INFO, WARNING, ERROR) with:               ║
# ║    - Console output (INFO and above by default)                            ║
# ║    - Rotating file output (DEBUG and above, max 10MB per file, 5 backups)  ║
# ║    - Structured format with timestamps, module, level, and message         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import logging
import logging.handlers
import os
import sys

# ── Constants ─────────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "transformer.log")
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB per log file
BACKUP_COUNT = 5                   # keep 5 rotated log files

# Structured format: timestamp | level | module | message
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Initializes the root logger with console and rotating-file handlers.

    Call this once at application startup (e.g., in train.py main()).
    All modules that use `logging.getLogger(__name__)` will inherit this config.

    Args:
        console_level: Minimum level for console output (default: INFO).
        file_level:    Minimum level for file output (default: DEBUG).
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Avoid adding duplicate handlers on repeated calls
    if root.handlers:
        return root

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler (INFO+) ───────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── Rotating file handler (DEBUG+) ────────────────────────────────────
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=MAX_LOG_BYTES, backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except (OSError, PermissionError) as e:
        # Fall back gracefully if we can't write to disk
        console.setLevel(logging.DEBUG)
        root.warning(f"Could not create log file at {LOG_FILE}: {e}. "
                     f"Logging to console only.")

    return root


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger for a specific module.

    Usage:
        from logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Model loaded successfully")
    """
    return logging.getLogger(name)
