"""In-memory ring buffer log handler for the dashboard."""

import logging
from collections import deque
from datetime import UTC, datetime

# Logger names to attach the ring buffer handler to
_LOGGER_NAMES = ("claudegate", "uvicorn", "uvicorn.error", "uvicorn.access")


class RingBufferHandler(logging.Handler):
    """Logging handler that stores recent log records in a fixed-size ring buffer.

    Thread-safe via deque's atomic append. Records are stored as dicts
    for easy JSON serialization.
    """

    def __init__(self, maxlen: int = 500):
        super().__init__()
        self._buffer: deque[dict] = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        self._buffer.append(
            {
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record),
            }
        )

    def get_entries(self, limit: int = 100, level_filter: str | None = None) -> list[dict]:
        """Return recent log entries, optionally filtered by level.

        Args:
            limit: Maximum number of entries to return.
            level_filter: If set, only return entries at or above this level
                          (e.g. "WARNING" returns WARNING, ERROR, CRITICAL).
        """
        entries = list(self._buffer)

        if level_filter:
            threshold = logging.getLevelName(level_filter.upper())
            if isinstance(threshold, int):
                entries = [e for e in entries if logging.getLevelName(e["level"]) >= threshold]

        return entries[-limit:]


# Module-level singleton
log_buffer = RingBufferHandler()


def attach_log_buffer() -> None:
    """Attach the ring buffer handler to all relevant loggers.

    Safe to call multiple times — skips loggers that already have the handler.
    Must be called after uvicorn's dictConfig runs (e.g. in the FastAPI lifespan)
    because uvicorn replaces all logger handlers on startup.
    """
    log_buffer.setFormatter(logging.Formatter("%(message)s"))
    for name in _LOGGER_NAMES:
        lgr = logging.getLogger(name)
        if log_buffer not in lgr.handlers:
            lgr.addHandler(log_buffer)
