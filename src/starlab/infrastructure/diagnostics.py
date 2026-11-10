"""Local startup evidence and native crash traces for intermittent desktop exits."""

import faulthandler
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from io import TextIOWrapper
from logging.handlers import RotatingFileHandler
from pathlib import Path


@contextmanager
def application_diagnostics(directory: Path) -> Iterator[logging.Logger]:
    """Record one desktop session without making an unwritable log directory fatal."""
    logger: logging.Logger = logging.getLogger("starlab")
    previous_level: int = logger.level
    with ExitStack() as resources:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            handler: RotatingFileHandler = RotatingFileHandler(
                directory / "startup.log", maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8"
            )
            resources.callback(handler.close)
            native_trace: TextIOWrapper = resources.enter_context(
                (directory / "native-crash.log").open("a", encoding="utf-8")
            )
        except OSError:
            logger.warning("Application diagnostics are unavailable in %s", directory)
            yield logger
            return
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        enable_native_trace: bool = not faulthandler.is_enabled()
        if enable_native_trace:
            native_trace.write(f"\nDesktop session PID {os.getpid()}\n")
            native_trace.flush()
            faulthandler.enable(file=native_trace)
        logger.info("Starting desktop session: PID %d; Python %s", os.getpid(), sys.version)
        try:
            yield logger
        except BaseException:
            logger.exception("Desktop session terminated with an uncaught exception")
            raise
        finally:
            logger.info("Desktop session finished")
            if enable_native_trace:
                faulthandler.disable()
            logger.removeHandler(handler)
            logger.setLevel(previous_level)
