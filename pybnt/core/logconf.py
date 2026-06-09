"""Logging configuration for PyBrainViewer."""

import logging

logger = logging.getLogger("pybnt")
logger.addHandler(logging.NullHandler())

def set_level(level: str = "WARNING") -> None:
    """Set the logging level for the PyBrainViewer logger."""
    logger.setLevel(getattr(logging, level.upper(), logging.WARNING))

def enable_console(level: str = "INFO") -> None:
    """Enable console output for the PyBrainViewer logger."""
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(name)s] %(levelname)s: %(message)s"
        ))
        logger.addHandler(handler)
