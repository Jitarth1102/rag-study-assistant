"""JSON logging utilities."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log_entry: dict[str, Any] = {
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        if record.__dict__:
            extra = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__}
            if extra:
                log_entry.update(extra)
        return json.dumps(log_entry)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[handler])


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_run_id() -> str:
    return str(uuid.uuid4())


__all__ = ["configure_logging", "get_logger", "get_run_id", "JsonFormatter"]
