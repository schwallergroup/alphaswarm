"""Logging module."""

import logging
import os
from datetime import date

import click
from rich.logging import RichHandler


class Logger:
    """Logging class."""

    def __init__(self, log_file: bool = False):
        """Initialise logger."""
        handlers = [
            RichHandler(markup=True, rich_tracebacks=True, tracebacks_suppress=[click])
        ]
        if log_file:
            os.makedirs("logs", exist_ok=True)
            file_name = date.today().strftime("%Y-%m-%d") + ".log"
            path = f"logs/{file_name}"
            handlers.append(logging.FileHandler(path, mode="a"))

        logging.basicConfig(
            level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=handlers
        )

        self.log = logging.getLogger("rich")
        self.log = logging.LoggerAdapter(self.log, {"markup": True})
