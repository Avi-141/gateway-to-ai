"""Shared exception types for backend error handling and fallback."""


class TransientBackendError(Exception):
    """Raised for backend errors eligible for fallback (429, 500, 502, 503, 504)."""

    def __init__(self, status_code: int, error_type: str, message: str, backend: str):
        self.status_code = status_code
        self.error_type = error_type
        self.message = message
        self.backend = backend
        super().__init__(f"{backend} error {status_code}: {message}")


class CopilotHttpError(Exception):
    """Non-transient Copilot HTTP error (401, 403, 404, etc.)."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Copilot HTTP {status_code}: {detail}")
