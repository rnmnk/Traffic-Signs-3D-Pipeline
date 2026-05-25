from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

import structlog

_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    *,
    file: str | Path | None = None,
    renderer: Literal["console", "json"] = "console",
) -> None:
    """Configure the logging system for the process."""
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level.upper())

    for h in list(root.handlers):
        root.removeHandler(h)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stream_handler)

    if file is not None:
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(file)
        fh.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(fh)

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.dev.ConsoleRenderer()
        if renderer == "console"
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger, configuring defaults if nothing was set yet."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name or "ts3d")
