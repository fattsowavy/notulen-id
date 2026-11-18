"""Logging setup shared by the CLI, the demo app, and scripts."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging with a consistent format."""
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, datefmt=DATE_FORMAT, force=True)
