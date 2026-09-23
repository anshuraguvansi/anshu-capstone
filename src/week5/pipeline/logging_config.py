import json
import logging
import time
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": round(time.time(), 3),
                "level": record.levelname,
                "msg": record.getMessage(),
                "logger": record.name,
            }
        )


def get_logger(
    name: str = "pipeline", logging_path: str | Path = "logs/pipeline.log"
) -> logging.Logger:
    """Returns a logger that writes JSON to the given path."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        return logger

    Path(logging_path).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(logging_path)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger
