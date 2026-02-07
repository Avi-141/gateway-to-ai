"""Anthropic <-> OpenAI format translation for GitHub Copilot backend.

Pure functions for request/response translation. No I/O, no state (except StreamTranslator).
"""

import json
import uuid
from typing import Any


# --- Request Translation (Anthropic -> OpenAI) ---


def _translate_content_to_openai(content: Any) -> str | list[dict[str, Any]]:
    """Translate Anthropic content blocks to OpenAI message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # If all blocks are text, join them
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block["text"])
                elif block.get("type") == "image":
                    # OpenAI vision format
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        texts.append(f"[image: {source.get('media_type', 'image')}]")
                elif block.get("type") == "tool_use":
                    # Handled separately in assistant messages
                    continue
                elif block.get("type") == "tool_result":
                    # Handled separately
                    continue
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else ""
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


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic tool definitions to OpenAI format."""
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
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
                        elif block.get("type") == "thinking":
                            # Drop thinking blocks - no direct OpenAI equivalent
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

    # Tools
    if "tools" in body:
        openai_body["tools"] = _translate_tools(body["tools"])
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


def openai_to_anthropic_response(
    openai_resp: dict[str, Any], model: str
) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response to Anthropic Messages format."""
    choice = openai_resp.get("choices", [{}])[0]
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
        },
    }


# --- Streaming Translation (OpenAI SSE -> Anthropic SSE) ---


class StreamTranslator:
    """Stateful translator converting OpenAI streaming chunks to Anthropic SSE events.

    State machine: INIT -> message_start -> content_block_start -> delta* ->
                   content_block_stop -> message_delta -> message_stop
    """

    def __init__(self, model: str):
        self.model = model
        self.block_index = 0
        self.started = False
        self.current_block_type: str | None = None
        # Track tool calls by index
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.has_text_block = False
        self.input_tokens = 0
        self.output_tokens = 0

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
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
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

    def _emit_content_block_stop(self) -> str:
        """Emit content_block_stop event."""
        event = self._sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.block_index},
        )
        self.block_index += 1
        self.current_block_type = None
        return event

    def translate_chunk(self, chunk: dict[str, Any]) -> str:
        """Translate a single OpenAI streaming chunk to Anthropic SSE events."""
        events = ""

        # Emit message_start on first chunk
        if not self.started:
            # Capture usage from first chunk if available
            usage = chunk.get("usage", {})
            if usage.get("prompt_tokens"):
                self.input_tokens = usage["prompt_tokens"]
            events += self._emit_message_start()

        choice = (chunk.get("choices") or [{}])[0]
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
                    events += self._emit_content_block_stop()

                # Start new tool_use block
                tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                tc_name = func.get("name", "")
                self.tool_calls[tc_index] = {
                    "id": tc_id,
                    "name": tc_name,
                    "arguments": "",
                }
                events += self._emit_content_block_start(
                    "tool_use", id=tc_id, name=tc_name
                )

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

        # Handle finish
        if finish_reason is not None:
            # Close current block if open
            if self.current_block_type is not None:
                events += self._emit_content_block_stop()

            # Capture final usage
            usage = chunk.get("usage", {})
            output_tokens = usage.get("completion_tokens", self.output_tokens)

            events += self._sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _translate_finish_reason(finish_reason),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": output_tokens},
                },
            )
            events += self._sse("message_stop", {"type": "message_stop"})

        return events
