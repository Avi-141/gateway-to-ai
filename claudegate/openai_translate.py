"""OpenAI <-> Anthropic format translation for OpenAI-compatible API surface.

Translates OpenAI Chat Completions requests to Anthropic Messages format (inbound)
and Anthropic responses back to OpenAI format (outbound). This enables OpenAI-format
clients (like Open WebUI) to use claudegate transparently.
"""

import json
import time
import uuid
from typing import Any

# --- Request Translation (OpenAI -> Anthropic) ---


def openai_to_anthropic_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions request to Anthropic Messages format."""
    anthropic_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "max_tokens": body.get("max_tokens") or 4096,
    }

    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []

    # Separate system messages and build Anthropic messages
    for msg in body.get("messages", []):
        role = msg.get("role", "user")

        if role == "system":
            # Collect system messages for Anthropic system field
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part["text"])
                    elif isinstance(part, str):
                        system_parts.append(part)

        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                # Assistant message with tool calls -> Anthropic content blocks
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    func = tc.get("function", {})
                    try:
                        arguments = json.loads(func.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": func.get("name", ""),
                            "input": arguments,
                        }
                    )
                messages.append({"role": "assistant", "content": blocks})
            else:
                messages.append({"role": "assistant", "content": content or ""})

        elif role == "tool":
            # Tool messages -> Anthropic user message with tool_result content block
            # Consecutive tool messages are merged into a single user message
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            # Check if last message is already a user message with tool_result blocks
            if messages and messages[-1].get("role") == "user" and isinstance(messages[-1].get("content"), list):
                last_content = messages[-1]["content"]
                if last_content and isinstance(last_content[0], dict) and last_content[0].get("type") == "tool_result":
                    last_content.append(tool_result)
                    continue
            messages.append({"role": "user", "content": [tool_result]})

        else:
            # Regular user message
            messages.append({"role": "user", "content": msg.get("content", "")})

    if system_parts:
        anthropic_body["system"] = "\n".join(system_parts)

    anthropic_body["messages"] = messages

    # Stream flag
    if "stream" in body:
        anthropic_body["stream"] = body["stream"]

    # Tools
    if "tools" in body:
        anthropic_body["tools"] = _translate_tools(body["tools"])
    if "tool_choice" in body:
        anthropic_body["tool_choice"] = _translate_tool_choice(body["tool_choice"])

    # Simple field mappings
    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]
    if "stop" in body:
        anthropic_body["stop_sequences"] = body["stop"]

    return anthropic_body


def _translate_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI tool definitions to Anthropic format."""
    anthropic_tools = []
    for tool in tools:
        func = tool.get("function", {})
        anthropic_tools.append(
            {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {}),
            }
        )
    return anthropic_tools


def _translate_tool_choice(tool_choice: Any) -> dict[str, Any]:
    """Translate OpenAI tool_choice to Anthropic format."""
    if isinstance(tool_choice, str):
        str_map = {"auto": "auto", "required": "any", "none": "none"}
        tc_type = str_map.get(tool_choice, "auto")
        return {"type": tc_type}
    elif isinstance(tool_choice, dict):
        # Specific function: {"type": "function", "function": {"name": "..."}}
        func = tool_choice.get("function", {})
        return {"type": "tool", "name": func.get("name", "")}
    return {"type": "auto"}


# --- Response Translation (Anthropic -> OpenAI) ---


def _translate_stop_reason(reason: str | None) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }
    return mapping.get(reason or "end_turn", "stop")


def anthropic_to_openai_response(anthropic_resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an Anthropic Messages response to OpenAI Chat Completions format."""
    content_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tc_index = 0

    for block in anthropic_resp.get("content", []):
        if block.get("type") == "text":
            content_text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                    "index": tc_index,
                }
            )
            tc_index += 1

    message: dict[str, Any] = {"role": "assistant"}
    content = "\n".join(content_text_parts) if content_text_parts else None
    message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = anthropic_resp.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:29]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _translate_stop_reason(anthropic_resp.get("stop_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# --- Streaming Translation (Anthropic SSE -> OpenAI SSE) ---


def parse_anthropic_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse Anthropic SSE text into a list of (event_type, data_dict) tuples.

    Handles the format: "event: TYPE\\ndata: JSON\\n\\n"
    """
    events: list[tuple[str, dict[str, Any]]] = []
    current_event_type = ""

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("event: "):
            current_event_type = line[7:].strip()
        elif line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                continue
            try:
                data = json.loads(data_str)
                events.append((current_event_type, data))
            except json.JSONDecodeError:
                continue

    return events


class ReverseStreamTranslator:
    """Stateful translator converting Anthropic SSE events to OpenAI streaming chunks.

    Processes Anthropic event types and emits OpenAI chat.completion.chunk objects.
    """

    def __init__(self, model: str):
        self.model = model
        self.msg_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        self.created = int(time.time())
        self.tool_call_index = 0
        self.sent_role = False
        self.current_block_type: str | None = None

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> str:
        """Format an OpenAI SSE chunk."""
        choice: dict[str, Any] = {"index": 0, "delta": delta}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        else:
            choice["finish_reason"] = None
        data = {
            "id": self.msg_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }
        return f"data: {json.dumps(data)}\n\n"

    def translate_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Translate a single Anthropic SSE event to OpenAI streaming chunk(s)."""
        chunks = ""

        if event_type == "message_start":
            # Emit role chunk
            if not self.sent_role:
                self.sent_role = True
                chunks += self._chunk({"role": "assistant", "content": ""})

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            self.current_block_type = block.get("type")
            if block.get("type") == "tool_use":
                # Emit tool call start chunk
                tool_call = {
                    "index": self.tool_call_index,
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": "",
                    },
                }
                chunks += self._chunk({"tool_calls": [tool_call]})

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                chunks += self._chunk({"content": delta.get("text", "")})
            elif delta.get("type") == "input_json_delta":
                tool_call = {
                    "index": self.tool_call_index,
                    "function": {
                        "arguments": delta.get("partial_json", ""),
                    },
                }
                chunks += self._chunk({"tool_calls": [tool_call]})

        elif event_type == "content_block_stop":
            if self.current_block_type == "tool_use":
                self.tool_call_index += 1
            self.current_block_type = None

        elif event_type == "message_delta":
            delta_data = data.get("delta", {})
            stop_reason = delta_data.get("stop_reason")
            finish_reason = _translate_stop_reason(stop_reason)
            chunks += self._chunk({}, finish_reason=finish_reason)

        elif event_type == "message_stop":
            # Emit [DONE]
            chunks += "data: [DONE]\n\n"

        return chunks

    def translate_sse(self, raw: str) -> str:
        """Translate raw Anthropic SSE text to OpenAI SSE text."""
        events = parse_anthropic_sse(raw)
        result = ""
        for event_type, data in events:
            result += self.translate_event(event_type, data)
        return result
