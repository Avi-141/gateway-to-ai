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


class ContextWindowExceededError(Exception):
    """Raised when the prompt exceeds the backend model's context window.

    Carries token counts so the proxy can return an error in the exact format
    that Claude Code recognises for auto-compaction.

    When exact token counts cannot be parsed, prompt_tokens and context_limit
    are 0 and raw_detail contains the original backend error message.
    """

    def __init__(self, prompt_tokens: int, context_limit: int, backend: str, raw_detail: str = ""):
        self.prompt_tokens = prompt_tokens
        self.context_limit = context_limit
        self.backend = backend
        self.raw_detail = raw_detail
        super().__init__(f"{backend}: prompt token count {prompt_tokens} exceeds context limit {context_limit}")
