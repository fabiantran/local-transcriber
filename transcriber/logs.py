from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "transcriber.log"
FORMAT = "%(asctime)s %(levelname)-7s %(message)s"


def configure_logging() -> logging.Logger:
    """Log processing metadata only (phases, durations, errors), never file names or text."""
    logger = logging.getLogger("transcriber")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)
    try:
        LOG_FILE.parent.mkdir(exist_ok=True)
        file = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    except OSError:
        logger.warning("Logdatei kann nicht geschrieben werden, nur Terminal-Ausgabe aktiv.")
    else:
        file.setFormatter(formatter)
        logger.addHandler(file)
    return logger
