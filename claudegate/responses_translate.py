"""Anthropic/OpenAI <-> Responses API format translation for codex models.

Codex models (gpt-5.x-codex) only support the /responses endpoint, not /chat/completions.
This module provides stateless translation functions and streaming state machines for
converting between Anthropic Messages / OpenAI Chat Completions and the Responses API format.
"""

import json
import time
import uuid
from typing import Any

# --- Request Translation (Anthropic -> Responses) ---


def anthropic_to_responses_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an Anthropic Messages API request to Responses API format."""
    responses_body: dict[str, Any] = {"model": model}

    # system -> instructions
    system = body.get("system")
    if system:
        if isinstance(system, str):
            responses_body["instructions"] = system
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
            if text_parts:
                responses_body["instructions"] = "\n".join(text_parts)

    # max_tokens -> max_output_tokens (Responses API requires >= 16)
    if "max_tokens" in body:
        responses_body["max_output_tokens"] = max(body["max_tokens"], 16)

    # Translate messages to input items
    input_items: list[dict[str, Any]] = []

    for msg in body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            # Tool results become top-level function_call_output items
                            tool_content = block.get("content", "")
                            if isinstance(tool_content, list):
                                text_parts = []
                                for tc in tool_content:
                                    if isinstance(tc, dict) and tc.get("type") == "text":
                                        text_parts.append(tc["text"])
                                tool_content = "\n".join(text_parts) if text_parts else ""
                            elif not isinstance(tool_content, str):
                                tool_content = str(tool_content) if tool_content else ""
                            input_items.append(
                                {
                                    "type": "function_call_output",
                                    "call_id": block.get("tool_use_id", ""),
                                    "output": tool_content,
                                }
                            )
                        elif block.get("type") == "text":
                            input_items.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": block["text"]}],
                                }
                            )
                    elif isinstance(block, str):
                        input_items.append(
                            {
                                "role": "user",
                                "content": [{"type": "input_text", "text": block}],
                            }
                        )
            elif isinstance(content, str):
                input_items.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}],
                    }
                )

        elif role == "assistant":
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_use":
                            # Flush any accumulated text
                            if text_parts:
                                input_items.append(
                                    {
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
                                    }
                                )
                                text_parts = []
                            input_items.append(
                                {
                                    "type": "function_call",
                                    "name": block.get("name", ""),
                                    "arguments": json.dumps(block.get("input", {})),
                                    "call_id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                }
                            )
                        elif block.get("type") == "text":
                            text_parts.append(block["text"])
                if text_parts:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
                        }
                    )
            elif isinstance(content, str) and content:
                input_items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )

    responses_body["input"] = input_items

    # Tools: input_schema -> parameters
    if "tools" in body:
        responses_body["tools"] = [
            {
                "type": "function",
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }
            for tool in body["tools"]
        ]

    # tool_choice
    if "tool_choice" in body:
        tc = body["tool_choice"]
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "auto":
                responses_body["tool_choice"] = "auto"
            elif tc_type == "any":
                responses_body["tool_choice"] = "required"
            elif tc_type == "tool":
                responses_body["tool_choice"] = {"type": "function", "name": tc.get("name", "")}
            elif tc_type == "none":
                responses_body["tool_choice"] = "none"

    # Simple field mappings
    if "temperature" in body:
        responses_body["temperature"] = body["temperature"]
    if "top_p" in body:
        responses_body["top_p"] = body["top_p"]

    return responses_body


# --- Request Translation (OpenAI Chat Completions -> Responses) ---


def openai_chat_to_responses_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions request to Responses API format."""
    responses_body: dict[str, Any] = {"model": model}

    input_items: list[dict[str, Any]] = []
    system_parts: list[str] = []

    for msg in body.get("messages", []):
        role = msg.get("role", "user")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        system_parts.append(part["text"])
                    elif isinstance(part, str):
                        system_parts.append(part)

        elif role == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                input_items.append(
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                if text_parts:
                    input_items.append(
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "\n".join(text_parts)}],
                        }
                    )

        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            if content:
                input_items.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    }
                )
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                            "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        }
                    )

        elif role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                }
            )

    if system_parts:
        responses_body["instructions"] = "\n".join(system_parts)

    responses_body["input"] = input_items

    # max_tokens -> max_output_tokens (Responses API requires >= 16)
    if "max_tokens" in body:
        responses_body["max_output_tokens"] = max(body["max_tokens"], 16)

    # Tools
    if "tools" in body:
        responses_body["tools"] = [
            {
                "type": "function",
                "name": tool.get("function", {}).get("name", ""),
                "description": tool.get("function", {}).get("description", ""),
                "parameters": tool.get("function", {}).get("parameters", {}),
            }
            for tool in body["tools"]
        ]

    # tool_choice
    if "tool_choice" in body:
        responses_body["tool_choice"] = body["tool_choice"]

    # Simple field mappings
    if "temperature" in body:
        responses_body["temperature"] = body["temperature"]
    if "top_p" in body:
        responses_body["top_p"] = body["top_p"]

    return responses_body


# --- Response Translation (Responses -> Anthropic) ---


def responses_to_anthropic_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a Responses API response to Anthropic Messages format."""
    content: list[dict[str, Any]] = []
    stop_reason = "end_turn"

    for item in resp.get("output", []):
        item_type = item.get("type")

        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content.append({"type": "text", "text": block.get("text", "")})

        elif item_type == "function_call":
            try:
                arguments = json.loads(item.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            content.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": item.get("name", ""),
                    "input": arguments,
                }
            )
            stop_reason = "tool_use"

        elif item_type == "reasoning":
            # Skip reasoning items
            continue

    if not content:
        content.append({"type": "text", "text": ""})

    # Check for explicit stop reason from response status
    status = resp.get("status")
    if status == "incomplete":
        incomplete_details = resp.get("incomplete_details", {})
        if incomplete_details.get("reason") == "max_output_tokens":
            stop_reason = "max_tokens"

    usage = resp.get("usage", {})

    return {
        "id": resp.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        },
    }


# --- Response Translation (Responses -> OpenAI Chat Completions) ---


def responses_to_openai_chat_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a Responses API response to OpenAI Chat Completions format."""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tc_index = 0
    finish_reason = "stop"

    for item in resp.get("output", []):
        item_type = item.get("type")

        if item_type == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content_parts.append(block.get("text", ""))

        elif item_type == "function_call":
            tool_calls.append(
                {
                    "id": item.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                    "index": tc_index,
                }
            )
            tc_index += 1
            finish_reason = "tool_calls"

        elif item_type == "reasoning":
            continue

    message: dict[str, Any] = {"role": "assistant"}
    message["content"] = "\n".join(content_parts) if content_parts else None
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp.get("usage", {})
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
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


# --- Streaming Translation (Responses SSE -> Anthropic SSE) ---


class ResponsesStreamTranslator:
    """Stateful translator converting Responses API SSE events to Anthropic SSE events.

    Responses SSE format uses event: + data: lines. Key events:
    - response.created -> message_start
    - response.output_text.delta -> content_block_delta (text)
    - response.output_item.added (function_call) -> content_block_start (tool_use)
    - response.function_call_arguments.delta -> content_block_delta (input_json)
    - response.output_item.done -> content_block_stop
    - response.completed -> message_delta + message_stop
    """

    def __init__(self, model: str, estimated_input_tokens: int = 0):
        self.model = model
        self.block_index = 0
        self.started = False
        self.current_block_type: str | None = None
        self.has_text_block = False
        self.input_tokens = estimated_input_tokens
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0

    def _sse(self, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _emit_message_start(self) -> str:
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
        event = self._sse(
            "content_block_stop",
            {"type": "content_block_stop", "index": self.block_index},
        )
        self.block_index += 1
        self.current_block_type = None
        return event

    def translate_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Translate a single Responses API SSE event to Anthropic SSE events."""
        if not isinstance(data, dict):
            return ""
        events = ""

        if event_type == "response.created":
            if not self.started:
                resp_obj = data.get("response") or {}
                usage = resp_obj.get("usage") or {}
                if usage.get("input_tokens"):
                    self.input_tokens = usage["input_tokens"]
                self.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
                self.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
                events += self._emit_message_start()

        elif event_type == "response.output_text.delta":
            if not self.started:
                events += self._emit_message_start()
            if not self.has_text_block:
                self.has_text_block = True
                events += self._emit_content_block_start("text")
            text = data.get("delta", "")
            events += self._sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        elif event_type == "response.output_item.added":
            if not self.started:
                events += self._emit_message_start()
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                # Close text block if open
                if self.has_text_block and self.current_block_type == "text":
                    events += self.emit_content_block_stop()
                call_id = item.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = item.get("name", "")
                events += self._emit_content_block_start("tool_use", id=call_id, name=name)

        elif event_type == "response.function_call_arguments.delta":
            delta = data.get("delta", "")
            if delta:
                events += self._sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self.block_index,
                        "delta": {"type": "input_json_delta", "partial_json": delta},
                    },
                )

        elif event_type == "response.output_item.done":
            if self.current_block_type is not None:
                events += self.emit_content_block_stop()

        elif event_type == "response.completed":
            # Close any open block
            if self.current_block_type is not None:
                events += self.emit_content_block_stop()

            resp_obj = data.get("response") or {}
            usage = resp_obj.get("usage") or {}
            self.output_tokens = usage.get("output_tokens", self.output_tokens)
            if usage.get("input_tokens"):
                self.input_tokens = usage["input_tokens"]
            if usage.get("cache_creation_input_tokens"):
                self.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
            if usage.get("cache_read_input_tokens"):
                self.cache_read_input_tokens = usage["cache_read_input_tokens"]

            # Determine stop reason
            status = resp_obj.get("status")
            if status == "incomplete":
                stop_reason = "max_tokens"
            else:
                # Check if we had tool calls
                output = resp_obj.get("output") or []
                has_tool = any(item.get("type") == "function_call" for item in output)
                stop_reason = "tool_use" if has_tool else "end_turn"

            events += self._sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {
                        "output_tokens": self.output_tokens,
                        "cache_creation_input_tokens": self.cache_creation_input_tokens,
                        "cache_read_input_tokens": self.cache_read_input_tokens,
                    },
                },
            )
            events += self._sse("message_stop", {"type": "message_stop"})

        return events


# --- Streaming Translation (Responses SSE -> OpenAI Chat Completions SSE) ---


class ResponsesToOpenAIStreamTranslator:
    """Stateful translator converting Responses API SSE events to OpenAI streaming chunks."""

    def __init__(self, model: str):
        self.model = model
        self.msg_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
        self.created = int(time.time())
        self.tool_call_index = 0
        self.sent_role = False

    def _chunk(self, delta: dict[str, Any], finish_reason: str | None = None) -> str:
        choice: dict[str, Any] = {"index": 0, "delta": delta}
        choice["finish_reason"] = finish_reason
        data = {
            "id": self.msg_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }
        return f"data: {json.dumps(data)}\n\n"

    def translate_event(self, event_type: str, data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        chunks = ""

        if event_type == "response.created":
            if not self.sent_role:
                self.sent_role = True
                chunks += self._chunk({"role": "assistant", "content": ""})

        elif event_type == "response.output_text.delta":
            if not self.sent_role:
                self.sent_role = True
                chunks += self._chunk({"role": "assistant", "content": ""})
            chunks += self._chunk({"content": data.get("delta", "")})

        elif event_type == "response.output_item.added":
            if not self.sent_role:
                self.sent_role = True
                chunks += self._chunk({"role": "assistant", "content": ""})
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                tool_call = {
                    "index": self.tool_call_index,
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": "",
                    },
                }
                chunks += self._chunk({"tool_calls": [tool_call]})

        elif event_type == "response.function_call_arguments.delta":
            delta = data.get("delta", "")
            if delta:
                tool_call = {
                    "index": self.tool_call_index,
                    "function": {"arguments": delta},
                }
                chunks += self._chunk({"tool_calls": [tool_call]})

        elif event_type == "response.output_item.done":
            item = data.get("item") or {}
            if item.get("type") == "function_call":
                self.tool_call_index += 1

        elif event_type == "response.completed":
            resp_obj = data.get("response") or {}
            output = resp_obj.get("output") or []
            has_tool = any(item.get("type") == "function_call" for item in output)
            if resp_obj.get("status") == "incomplete":
                finish_reason = "length"
            elif has_tool:
                finish_reason = "tool_calls"
            else:
                finish_reason = "stop"
            chunks += self._chunk({}, finish_reason=finish_reason)
            chunks += "data: [DONE]\n\n"

        return chunks
