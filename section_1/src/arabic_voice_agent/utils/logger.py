"""Consistent, readable application logging."""

import logging

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the application logger exactly once."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("arabic_voice_agent")


logger = configure_logging()
