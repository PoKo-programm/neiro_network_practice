import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Настраивает логгер с выводом в stderr и заданным уровнем."""
    logger = logging.getLogger("dataprep")
    logger.setLevel(level)
    logger.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
