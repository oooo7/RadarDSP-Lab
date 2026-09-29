"""Logging configuration for RadarDSP Lab."""

import logging
import sys


def setup_logger(name: str = "radar_dsp_lab", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standard logger for the application.

    Args:
        name: Logger module name.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
