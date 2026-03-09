"""Anthropic <-> OpenAI format translation for GitHub Copilot backend.

Stateless translation functions (except StreamTranslator) and input token estimation
via tiktoken. No network I/O.
"""

import json
import uuid
from typing import Any

import tiktoken

# Lazy-initialized tokenizer for estimating input tokens
_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Return the shared tokenizer, initializing on first use."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def estimate_input_tokens(body: dict[str, Any]) -> int:
    """Estimate input token count from an Anthropic Messages request body.

    Uses cl100k_base tokenizer (close to Claude's tokenizer) to approximate
    the input token count. This is used as a fallback when the backend
    doesn't provide prompt_tokens (e.g. OpenAI streaming).
    """
    tokenizer = _get_tokenizer()
    total = 0

    # System prompt
    system = body.get("system")
    if isinstance(system, str):
        total += len(tokenizer.encode(system))
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                total += len(tokenizer.encode(block.get("text", "")))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(tokenizer.encode(content))
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        total += len(tokenizer.encode(text))
                elif block_type == "tool_use":
                    total += len(tokenizer.encode(block.get("name", "")))
                    total += len(tokenizer.encode(json.dumps(block.get("input", {}))))
                elif block_type == "image":
                    total += 1600  # rough estimate per image
                elif block_type == "tool_result":
                    tc = block.get("content", "")
                    if isinstance(tc, str):
                        total += len(tokenizer.encode(tc))
                    elif isinstance(tc, list):
                        for sub in tc:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                total += len(tokenizer.encode(sub.get("text", "")))

    # Tool definitions
    tools = body.get("tools")
    if tools:
        total += len(tokenizer.encode(json.dumps(tools)))

    return total


# --- Request Translation (Anthropic -> OpenAI) ---


def _translate_content_to_openai(content: Any) -> str | list[dict[str, Any]]:
    """Translate Anthropic content blocks to OpenAI message content.

    Returns a string when the content is text-only, or a list of content parts
    when images are present (OpenAI vision format).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        image_parts: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{media_type};base64,{data}"},
                            }
                        )
                    elif source.get("type") == "url":
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": source.get("url", "")},
                            }
                        )
                elif block.get("type") == "tool_use":
                    # Handled separately in assistant messages
                    continue
                elif block.get("type") == "tool_result":
                    # Handled separately
                    continue
            elif isinstance(block, str):
                text_parts.append(block)
        if image_parts:
            # Return list of content parts for OpenAI vision format
            parts: list[dict[str, Any]] = []
            for text in text_parts:
                parts.append({"type": "text", "text": text})
            parts.extend(image_parts)
            return parts
        return "\n".join(text_parts) if text_parts else ""
    return str(content) if content else ""


def _translate_tool_choice(tool_choice: Any) -> Any:
    """Translate Anthropic tool_choice to OpenAI format."""
    if not isinstance(tool_choice, dict):
        return tool_choice
    tc_type = tool_choice.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    elif tc_type == "any":
        return "required"
    elif tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    elif tc_type == "none":
        return "none"
    return "auto"


def _is_server_tool(tool: dict[str, Any]) -> bool:
    """Check if a tool is a server-side tool (e.g. web_search).

    Server-side tools have a type field that is NOT "custom" (the default
    for regular function-calling tools). They are executed by the API server,
    not the client, so they cannot be translated to OpenAI function-calling format.
    """
    return tool.get("type", "custom") != "custom"


def has_server_tools(body: dict[str, Any]) -> bool:
    """Check if a request body contains any server-side tools."""
    return any(_is_server_tool(tool) for tool in body.get("tools", []))


def strip_server_tools(body: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the request body with server-side tools removed.

    Also strips server_tool_use and web_search_tool_result content blocks
    from conversation history, and removes tools/tool_choice if no tools remain.
    """
    body = {**body}

    # Filter tools
    if "tools" in body:
        client_tools = [t for t in body["tools"] if not _is_server_tool(t)]
        if client_tools:
            body["tools"] = client_tools
        else:
            del body["tools"]
            body.pop("tool_choice", None)

    # Strip server tool content blocks from messages
    if "messages" in body:
        cleaned_messages = []
        for msg in body["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                cleaned_blocks = [
                    block
                    for block in content
                    if not isinstance(block, dict)
                    or block.get("type") not in ("server_tool_use", "web_search_tool_result")
                ]
                if cleaned_blocks:
                    cleaned_messages.append({**msg, "content": cleaned_blocks})
                # Drop messages that become empty after stripping
            else:
                cleaned_messages.append(msg)
        body["messages"] = cleaned_messages

    return body


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions to OpenAI format.

    Skips server-side tools (e.g. web_search) which cannot be converted
    to OpenAI function-calling format.
    """
    openai_tools = []
    for tool in tools:
        if _is_server_tool(tool):
            continue
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", "") or "No description provided.",
                "parameters": tool.get("input_schema", {}),
            },
        }
        openai_tools.append(openai_tool)
    return openai_tools


def anthropic_to_openai_request(body: dict[str, Any], openai_model: str) -> dict[str, Any]:
    """Translate an Anthropic Messages API request to OpenAI Chat Completions format."""
    openai_body: dict[str, Any] = {
        "model": openai_model,
        "max_tokens": body.get("max_tokens", 4096),
    }

    messages: list[dict[str, Any]] = []

    # System prompt -> system message
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic allows system as list of content blocks
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # Translate messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "assistant":
            # Check for tool_use blocks
            tool_calls = []
            text_parts = []

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            tool_calls.append(
                                {
                                    "id": block.get("id", str(uuid.uuid4())),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input", {})),
                                    },
                                }
                            )
                        elif block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") in ("thinking", "server_tool_use"):
                            # Drop thinking/server_tool_use blocks - no OpenAI equivalent
                            continue
            elif isinstance(content, str):
                text_parts.append(content)

            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                assistant_msg["content"] = "\n".join(text_parts)
            else:
                assistant_msg["content"] = None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

        elif role == "user":
            # Check for tool_result blocks
            if isinstance(content, list):
                # Separate tool_results from regular content
                regular_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # Each tool_result becomes a separate tool message
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            tool_text_parts = []
                            for tc in tool_content:
                                if isinstance(tc, dict) and tc.get("type") == "text":
                                    tool_text_parts.append(tc["text"])
                            tool_content = "\n".join(tool_text_parts) if tool_text_parts else ""
                        elif not isinstance(tool_content, str):
                            tool_content = str(tool_content) if tool_content else ""
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": tool_content,
                            }
                        )
                    elif isinstance(block, dict) and block.get("type") == "web_search_tool_result":
                        # Drop web_search_tool_result blocks - server-side tool result
                        continue
                    else:
                        regular_parts.append(block)

                if regular_parts:
                    messages.append(
                        {
                            "role": "user",
                            "content": _translate_content_to_openai(regular_parts),
                        }
                    )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": _translate_content_to_openai(content),
                    }
                )
        else:
            messages.append({"role": role, "content": _translate_content_to_openai(content)})

    openai_body["messages"] = messages

    # Tools (skip if all tools are server-side and got filtered out)
    if "tools" in body:
        translated = _translate_tools(body["tools"])
        if translated:
            openai_body["tools"] = translated
            if "tool_choice" in body:
                openai_body["tool_choice"] = _translate_tool_choice(body["tool_choice"])

    # Simple field mappings
    if "temperature" in body:
        openai_body["temperature"] = body["temperature"]
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        openai_body["stop"] = body["stop_sequences"]

    # stream flag handled by caller, not in body
    if "stream" in body:
        openai_body["stream"] = body["stream"]

    return openai_body


# --- Response Translation (OpenAI -> Anthropic) ---


def _translate_finish_reason(reason: str | None) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(reason or "stop", "end_turn")


def openai_to_anthropic_response(openai_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response to Anthropic Messages format."""
    choices = openai_resp.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    content: list[dict[str, Any]] = []

    # Text content
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})

    # Tool calls
    for tc in message.get("tool_calls") or []:
        func = tc.get("function", {})
        try:
            arguments = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        content.append(
            {
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": func.get("name", ""),
                "input": arguments,
            }
        )

    if not content:
        content.append({"type": "text", "text": ""})

    # Usage
    usage = openai_resp.get("usage", {})

    return {
        "id": openai_resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _translate_finish_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        },
    }


# --- Streaming Translation (OpenAI SSE -> Anthropic SSE) ---


class StreamTranslator:
    """Stateful translator converting OpenAI streaming chunks to Anthropic SSE events.

    State machine: INIT -> message_start -> content_block_start -> delta* ->
                   content_block_stop -> message_delta -> message_stop
    """

    def __init__(
        self,
        model: str,
        estimated_input_tokens: int = 0,
        copilot_context_limit: int = 0,
        client_context_window: int = 0,
    ):
        self.model = model
        self.block_index = 0
        self.started = False
        self.current_block_type: str | None = None
        # Track tool calls by index
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.has_text_block = False
        self._copilot_context_limit = copilot_context_limit
        self._client_context_window = client_context_window
        self.input_tokens = self._scale_tokens(estimated_input_tokens)
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self._finish_reason: str | None = None
        self._finished = False

    def _scale_tokens(self, tokens: int) -> int:
        """Scale token count from Copilot's context window to the client's expected window.

        When Copilot enforces a smaller prompt limit than the client expects,
        raw token counts make usage appear lower than reality. Scaling corrects this
        so Claude Code's percentage-based context tracking remains accurate.

        The client_context_window is determined per-request: 1M when the anthropic-beta
        header contains 'context-1m', otherwise the model's max_context_window_tokens
        from the Copilot API (typically 200k).
        """
        if self._copilot_context_limit > 0 and self._client_context_window > 0:
            return int(tokens * self._client_context_window / self._copilot_context_limit)
        return tokens

    def _sse(self, event_type: str, data: dict[str, Any]) -> str:
        """Format an Anthropic SSE event."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _emit_message_start(self) -> str:
        """Emit the initial message_start event."""
        self.started = True
        return self._sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": self.cache_creation_input_tokens,
                        "cache_read_input_tokens": self.cache_read_input_tokens,
                    },
                },
            },
        )

    def _emit_content_block_start(self, block_type: str, **kwargs: Any) -> str:
        """Emit content_block_start event."""
        self.current_block_type = block_type
        block: dict[str, Any] = {"type": block_type}
        if block_type == "text":
            block["text"] = ""
        elif block_type == "tool_use":
            block["id"] = kwargs.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
            block["name"] = kwargs.get("name", "")
            block["input"] = {}
        return self._sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self.block_index,
                "content_block": block,
            },
        )

    def emit_content_block_stop(self) -> str:
        """Emit content_block_stop event."""
        event = self._sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.block_index},
        )
        self.block_index += 1
        self.current_block_type = None
        return event

    def translate_chunk(self, chunk: dict[str, Any]) -> str:
        """Translate a single OpenAI streaming chunk to Anthropic SSE events.

        With stream_options.include_usage, OpenAI sends chunks in this order:
        1. Content chunks (choices with deltas, no usage)
        2. Final content chunk (with finish_reason in choices)
        3. Usage-only chunk (choices: [], usage: {...})

        We defer emitting message_delta/message_stop until we receive usage data
        so that Claude Code gets accurate input_tokens for context tracking.
        """
        events = ""

        # Emit message_start on first chunk
        if not self.started:
            # Capture usage from first chunk if available (unlikely in streaming)
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = self._scale_tokens(usage["prompt_tokens"])
            if usage.get("cache_creation_input_tokens") is not None:
                self.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                self.cache_read_input_tokens = usage["cache_read_input_tokens"]
            events += self._emit_message_start()

        choices = chunk.get("choices") or []
        usage = chunk.get("usage") or {}

        # Handle usage-only chunk (final chunk with include_usage)
        if not choices and usage:
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = self._scale_tokens(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                self.output_tokens = self._scale_tokens(usage["completion_tokens"])
            if usage.get("cache_creation_input_tokens") is not None:
                self.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                self.cache_read_input_tokens = usage["cache_read_input_tokens"]
            # If we already saw finish_reason, emit the deferred message_delta/stop now
            if self._finish_reason is not None and not self._finished:
                events += self._emit_message_end()
            return events

        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle text content
        text_content = delta.get("content")
        if text_content is not None:
            if not self.has_text_block:
                self.has_text_block = True
                events += self._emit_content_block_start("text")
            events += self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text_content},
                },
            )

        # Handle tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            func = tc.get("function", {})

            if tc_index not in self.tool_calls:
                # New tool call - close text block if open
                if self.has_text_block and self.current_block_type == "text":
                    events += self.emit_content_block_stop()

                # Start new tool_use block
                tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                tc_name = func.get("name", "")
                self.tool_calls[tc_index] = {
                    "id": tc_id,
                    "name": tc_name,
                    "arguments": "",
                }
                events += self._emit_content_block_start("tool_use", id=tc_id, name=tc_name)

            # Accumulate arguments
            if func.get("arguments"):
                self.tool_calls[tc_index]["arguments"] += func["arguments"]
                events += self._sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": func["arguments"],
                        },
                    },
                )

        # Handle finish_reason
        if finish_reason is not None:
            # Close current block if open
            if self.current_block_type is not None:
                events += self.emit_content_block_stop()

            self._finish_reason = finish_reason

            # Capture usage from this chunk if present
            if usage.get("completion_tokens") is not None:
                self.output_tokens = self._scale_tokens(usage["completion_tokens"])
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = self._scale_tokens(usage["prompt_tokens"])

            # If usage was already provided in this chunk (no separate usage chunk),
            # emit message_delta/stop immediately
            if usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None:
                events += self._emit_message_end()

        return events

    def _emit_message_end(self) -> str:
        """Emit message_delta and message_stop events."""
        if self._finished:
            return ""
        self._finished = True
        events = self._sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _translate_finish_reason(self._finish_reason),
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": self.output_tokens,
                    "cache_creation_input_tokens": self.cache_creation_input_tokens,
                    "cache_read_input_tokens": self.cache_read_input_tokens,
                },
            },
        )
        events += self._sse("message_stop", {"type": "message_stop"})
        return events

    def flush(self) -> str:
        """Flush any pending state. Call after stream ends.

        If finish_reason was received but no usage-only chunk followed,
        emit message_delta/stop with whatever usage we have (including estimates).
        """
        events = ""
        if self.current_block_type is not None:
            events += self.emit_content_block_stop()
        if self._finish_reason is not None and not self._finished:
            events += self._emit_message_end()
        return events
