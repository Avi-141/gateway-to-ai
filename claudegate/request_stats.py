"""In-memory request statistics tracking."""


class RequestStats:
    """Simple in-memory counter for request metrics.

    Thread-safe for integer increments under the GIL.
    Ephemeral — resets on restart, no persistence.
    """

    def __init__(self) -> None:
        self.total_requests: int = 0
        self.requests_by_backend: dict[str, int] = {}
        self.errors: int = 0
        self.fallbacks: int = 0
        self.context_guard_rejections: int = 0

    def record_request(self, backend: str) -> None:
        """Record a request routed to the given backend."""
        self.total_requests += 1
        self.requests_by_backend[backend] = self.requests_by_backend.get(backend, 0) + 1

    def record_error(self) -> None:
        """Record a request that resulted in an error response."""
        self.errors += 1

    def record_fallback(self) -> None:
        """Record a fallback from primary to secondary backend."""
        self.fallbacks += 1

    def record_context_guard_rejection(self) -> None:
        """Record a request rejected by the pre-flight context guard."""
        self.context_guard_rejections += 1

    def snapshot(self) -> dict:
        """Return a serializable dict of current stats."""
        return {
            "total_requests": self.total_requests,
            "requests_by_backend": dict(self.requests_by_backend),
            "errors": self.errors,
            "fallbacks": self.fallbacks,
            "context_guard_rejections": self.context_guard_rejections,
        }

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_requests = 0
        self.requests_by_backend.clear()
        self.errors = 0
        self.fallbacks = 0
        self.context_guard_rejections = 0


# Module-level singleton
request_stats = RequestStats()
