"""Anthropic/OpenAI <-> Responses API format translation.

Provides bidirectional translation between the Responses API format and both
Anthropic Messages and OpenAI Chat Completions formats. Used for:
- Internal routing: models that only support /responses get translated from/to other formats
- Client-facing /v1/responses endpoint: incoming Responses requests get translated to Anthropic/OpenAI
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
                            user_content_parts: list[dict[str, Any]] = [{"type": "input_text", "text": block["text"]}]
                            input_items.append(
                                {
                                    "role": "user",
                                    "content": user_content_parts,
                                }
                            )
                        elif block.get("type") == "image":
                            source = block.get("source", {})
                            if source.get("type") == "base64":
                                media_type = source.get("media_type", "image/png")
                                data = source.get("data", "")
                                image_url = f"data:{media_type};base64,{data}"
                            else:
                                image_url = source.get("url", "")
                            input_items.append(
                                {
                                    "role": "user",
                                    "content": [{"type": "input_image", "image_url": image_url}],
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
                content_parts: list[dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append({"type": "input_text", "text": part.get("text", "")})
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "") if isinstance(image_url, dict) else ""
                        content_parts.append({"type": "input_image", "image_url": url})
                    elif isinstance(part, str):
                        content_parts.append({"type": "input_text", "text": part})
                if content_parts:
                    input_items.append(
                        {
                            "role": "user",
                            "content": content_parts,
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

    def flush(self) -> str:
        """Close any open content block. Call after stream ends."""
        if self.current_block_type is not None:
            return self.emit_content_block_stop()
        return ""

    def translate_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Translate a single Responses API SSE event to Anthropic SSE events."""
        if not isinstance(data, dict):
            return ""
        events = ""

        if event_type == "response.created":
            if not self.started:
                resp_obj = data.get("response") or {}
                usage = resp_obj.get("usage") or {}
                if usage.get("input_tokens") is not None:
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
            if usage.get("input_tokens") is not None:
                self.input_tokens = usage["input_tokens"]
            if usage.get("cache_creation_input_tokens") is not None:
                self.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
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


# --- Reverse Request Translation (Responses -> Anthropic) ---


def responses_to_anthropic_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate a Responses API request to Anthropic Messages format."""
    anthropic_body: dict[str, Any] = {
        "model": body.get("model", ""),
        "max_tokens": body.get("max_output_tokens") or 4096,
    }

    # instructions -> system
    instructions = body.get("instructions")
    if instructions:
        anthropic_body["system"] = instructions

    # Translate input items to messages
    messages: list[dict[str, Any]] = []
    input_items = body.get("input", [])

    # Handle string input (simple prompt)
    if isinstance(input_items, str):
        messages.append({"role": "user", "content": input_items})
        anthropic_body["messages"] = messages
        return _finalize_responses_to_anthropic(anthropic_body, body)

    # Group consecutive function_calls into a single assistant message
    i = 0
    while i < len(input_items):
        item = input_items[i]

        if isinstance(item, dict):
            item_type = item.get("type")
            role = item.get("role")

            if role == "user":
                content = item.get("content", [])
                # Convert input_text and input_image blocks
                anthropic_blocks: list[dict[str, Any]] = []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "input_text":
                            anthropic_blocks.append({"type": "text", "text": block.get("text", "")})
                        elif isinstance(block, dict) and block.get("type") == "input_image":
                            image_url = block.get("image_url", "")
                            if isinstance(image_url, str) and image_url.startswith("data:"):
                                header, _, data = image_url.partition(",")
                                media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                                anthropic_blocks.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "base64", "media_type": media_type, "data": data},
                                    }
                                )
                            elif isinstance(image_url, str) and image_url:
                                anthropic_blocks.append(
                                    {
                                        "type": "image",
                                        "source": {"type": "url", "url": image_url},
                                    }
                                )
                        elif isinstance(block, str):
                            anthropic_blocks.append({"type": "text", "text": block})
                elif isinstance(content, str):
                    anthropic_blocks.append({"type": "text", "text": content})
                if anthropic_blocks:
                    # Use list content to preserve image blocks
                    if len(anthropic_blocks) == 1 and anthropic_blocks[0].get("type") == "text":
                        messages.append({"role": "user", "content": anthropic_blocks[0]["text"]})
                    else:
                        messages.append({"role": "user", "content": anthropic_blocks})

            elif role == "assistant":
                content = item.get("content", [])
                text_parts = []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            text_parts.append(block.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)
                if text_parts:
                    messages.append({"role": "assistant", "content": "\n".join(text_parts)})

            elif item_type == "function_call":
                # Collect consecutive function_calls into one assistant message
                tool_use_blocks: list[dict[str, Any]] = []
                while i < len(input_items):
                    fc = input_items[i]
                    if not isinstance(fc, dict) or fc.get("type") != "function_call":
                        break
                    try:
                        arguments = json.loads(fc.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        arguments = {}
                    tool_use_blocks.append(
                        {
                            "type": "tool_use",
                            "id": fc.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": fc.get("name", ""),
                            "input": arguments,
                        }
                    )
                    i += 1
                messages.append({"role": "assistant", "content": tool_use_blocks})
                continue  # skip i += 1 at end

            elif item_type == "function_call_output":
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                }
                # Merge consecutive tool results into a single user message
                if (
                    messages
                    and messages[-1].get("role") == "user"
                    and isinstance(messages[-1].get("content"), list)
                    and messages[-1]["content"]
                    and isinstance(messages[-1]["content"][0], dict)
                    and messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    messages[-1]["content"].append(tool_result)
                else:
                    messages.append({"role": "user", "content": [tool_result]})

        i += 1

    anthropic_body["messages"] = messages
    return _finalize_responses_to_anthropic(anthropic_body, body)


def _finalize_responses_to_anthropic(anthropic_body: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Add tools and simple field mappings to the Anthropic request."""
    # Tools: function type -> Anthropic format (parameters -> input_schema)
    if "tools" in body:
        anthropic_tools = []
        for tool in body["tools"]:
            if tool.get("type") == "function":
                anthropic_tools.append(
                    {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters", {}),
                    }
                )
        if anthropic_tools:
            anthropic_body["tools"] = anthropic_tools

    # tool_choice
    if "tool_choice" in body:
        tc = body["tool_choice"]
        if isinstance(tc, str):
            str_map = {"auto": "auto", "required": "any", "none": "none"}
            tc_type = str_map.get(tc, "auto")
            anthropic_body["tool_choice"] = {"type": tc_type}
        elif isinstance(tc, dict):
            if tc.get("type") == "function":
                anthropic_body["tool_choice"] = {"type": "tool", "name": tc.get("name", "")}

    # Simple field mappings
    if "temperature" in body:
        anthropic_body["temperature"] = body["temperature"]
    if "top_p" in body:
        anthropic_body["top_p"] = body["top_p"]

    return anthropic_body


# --- Reverse Request Translation (Responses -> OpenAI Chat Completions) ---


def responses_to_openai_chat_request(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate a Responses API request to OpenAI Chat Completions format."""
    openai_body: dict[str, Any] = {"model": model}

    messages: list[dict[str, Any]] = []

    # instructions -> system message
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # max_output_tokens -> max_tokens
    if "max_output_tokens" in body:
        openai_body["max_tokens"] = body["max_output_tokens"]

    input_items = body.get("input", [])

    # Handle string input
    if isinstance(input_items, str):
        messages.append({"role": "user", "content": input_items})
        openai_body["messages"] = messages
        return _finalize_responses_to_openai_chat(openai_body, body)

    i = 0
    while i < len(input_items):
        item = input_items[i]

        if isinstance(item, dict):
            item_type = item.get("type")
            role = item.get("role")

            if role == "user":
                content = item.get("content", [])
                openai_parts: list[dict[str, Any]] = []
                text_parts: list[str] = []
                has_images = False
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "input_text":
                            text_parts.append(block.get("text", ""))
                            openai_parts.append({"type": "text", "text": block.get("text", "")})
                        elif isinstance(block, dict) and block.get("type") == "input_image":
                            has_images = True
                            openai_parts.append({"type": "image_url", "image_url": {"url": block.get("image_url", "")}})
                        elif isinstance(block, str):
                            text_parts.append(block)
                            openai_parts.append({"type": "text", "text": block})
                elif isinstance(content, str):
                    text_parts.append(content)
                if has_images:
                    messages.append({"role": "user", "content": openai_parts})
                elif text_parts:
                    messages.append({"role": "user", "content": "\n".join(text_parts)})

            elif role == "assistant":
                content = item.get("content", [])
                text_parts = []
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "output_text":
                            text_parts.append(block.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)
                if text_parts:
                    messages.append({"role": "assistant", "content": "\n".join(text_parts)})

            elif item_type == "function_call":
                # Collect consecutive function_calls into a single assistant message with tool_calls
                tool_calls: list[dict[str, Any]] = []
                while i < len(input_items):
                    fc = input_items[i]
                    if not isinstance(fc, dict) or fc.get("type") != "function_call":
                        break
                    tool_calls.append(
                        {
                            "id": fc.get("call_id", f"call_{uuid.uuid4().hex[:24]}"),
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": fc.get("arguments", "{}"),
                            },
                        }
                    )
                    i += 1
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
                assistant_msg["tool_calls"] = tool_calls
                messages.append(assistant_msg)
                continue

            elif item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )

        i += 1

    openai_body["messages"] = messages
    return _finalize_responses_to_openai_chat(openai_body, body)


def _finalize_responses_to_openai_chat(openai_body: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """Add tools and simple field mappings to the OpenAI Chat Completions request."""
    if "tools" in body:
        openai_tools = []
        for tool in body["tools"]:
            if tool.get("type") == "function":
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.get("name", ""),
                            "description": tool.get("description", ""),
                            "parameters": tool.get("parameters", {}),
                        },
                    }
                )
        if openai_tools:
            openai_body["tools"] = openai_tools

    if "tool_choice" in body:
        openai_body["tool_choice"] = body["tool_choice"]

    if "temperature" in body:
        openai_body["temperature"] = body["temperature"]
    if "top_p" in body:
        openai_body["top_p"] = body["top_p"]

    return openai_body


# --- Reverse Response Translation (Anthropic -> Responses) ---


def anthropic_to_responses_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an Anthropic Messages response to Responses API format."""
    output: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for block in resp.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            # Flush accumulated text into a message output item
            if text_parts:
                output.append(
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
                    }
                )
                text_parts = []
            output.append(
                {
                    "type": "function_call",
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                    "call_id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                }
            )

    # Flush remaining text
    if text_parts:
        output.append(
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
            }
        )

    # Map stop_reason to status
    stop_reason = resp.get("stop_reason", "end_turn")
    status = "incomplete" if stop_reason == "max_tokens" else "completed"

    usage = resp.get("usage", {})

    result: dict[str, Any] = {
        "id": resp.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }

    # Include cache tokens if present
    if usage.get("cache_creation_input_tokens"):
        result["usage"]["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]
    if usage.get("cache_read_input_tokens"):
        result["usage"]["cache_read_input_tokens"] = usage["cache_read_input_tokens"]

    if status == "incomplete":
        result["incomplete_details"] = {"reason": "max_output_tokens"}

    return result


# --- Reverse Response Translation (OpenAI Chat Completions -> Responses) ---


def openai_chat_to_responses_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an OpenAI Chat Completions response to Responses API format."""
    choice = resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    output: list[dict[str, Any]] = []

    # Text content
    if message.get("content"):
        output.append(
            {
                "type": "message",
                "content": [{"type": "output_text", "text": message["content"]}],
            }
        )

    # Tool calls
    for tc in message.get("tool_calls") or []:
        func = tc.get("function", {})
        output.append(
            {
                "type": "function_call",
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            }
        )

    # Map finish_reason to status
    finish_reason = choice.get("finish_reason", "stop")
    status = "incomplete" if finish_reason == "length" else "completed"

    usage = resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)

    result: dict[str, Any] = {
        "id": resp.get("id", f"resp_{uuid.uuid4().hex[:24]}"),
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }

    if status == "incomplete":
        result["incomplete_details"] = {"reason": "max_output_tokens"}

    return result


# --- Streaming Translation (Anthropic SSE -> Responses SSE) ---


class AnthropicToResponsesStreamTranslator:
    """Stateful translator converting Anthropic SSE events to Responses SSE events.

    Anthropic event types mapped to Responses events:
    - message_start -> response.created
    - content_block_start (text) -> response.output_item.added (message)
    - content_block_delta (text_delta) -> response.output_text.delta
    - content_block_start (tool_use) -> response.output_item.added (function_call)
    - content_block_delta (input_json_delta) -> response.function_call_arguments.delta
    - content_block_stop -> response.output_item.done
    - message_delta -> accumulate stop_reason/usage
    - message_stop -> response.completed
    """

    def __init__(self, model: str):
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.output_items: list[dict[str, Any]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_creation_input_tokens = 0
        self.cache_read_input_tokens = 0
        self.stop_reason: str | None = None
        self._current_text_parts: list[str] = []
        self._current_tool: dict[str, Any] | None = None
        self._current_tool_args: str = ""
        self._started = False
        self._finished = False
        self._item_index = 0

    def _sse(self, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def translate_event(self, event_type: str, data: dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        events = ""

        if event_type == "message_start":
            if not self._started:
                self._started = True
                message = data.get("message", {})
                usage = message.get("usage", {})
                self.input_tokens = usage.get("input_tokens", 0)
                self.cache_creation_input_tokens = usage.get("cache_creation_input_tokens", 0)
                self.cache_read_input_tokens = usage.get("cache_read_input_tokens", 0)
                events += self._sse(
                    "response.created",
                    {
                        "type": "response.created",
                        "response": {
                            "id": self.response_id,
                            "object": "response",
                            "created_at": self.created_at,
                            "status": "in_progress",
                            "model": self.model,
                            "output": [],
                            "usage": {
                                "input_tokens": self.input_tokens,
                                "output_tokens": 0,
                                "total_tokens": self.input_tokens,
                            },
                        },
                    },
                )

        elif event_type == "content_block_start":
            block = data.get("content_block", {})
            block_type = block.get("type")

            if block_type == "text":
                self._current_text_parts = []
                item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                }
                events += self._sse(
                    "response.output_item.added",
                    {"type": "response.output_item.added", "output_index": self._item_index, "item": item},
                )

            elif block_type == "tool_use":
                self._current_tool = {
                    "type": "function_call",
                    "name": block.get("name", ""),
                    "arguments": "",
                    "call_id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                }
                self._current_tool_args = ""
                events += self._sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": self._item_index,
                        "item": self._current_tool,
                    },
                )

        elif event_type == "content_block_delta":
            delta = data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                text = delta.get("text", "")
                self._current_text_parts.append(text)
                events += self._sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "output_index": self._item_index,
                        "content_index": 0,
                        "delta": text,
                    },
                )

            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json", "")
                self._current_tool_args += partial
                events += self._sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": self._item_index,
                        "delta": partial,
                    },
                )

        elif event_type == "content_block_stop":
            # Build done item
            if self._current_tool is not None:
                done_item = {**self._current_tool, "arguments": self._current_tool_args}
                self._current_tool = None
                self._current_tool_args = ""
            else:
                done_item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "".join(self._current_text_parts)}],
                }
                self._current_text_parts = []

            self.output_items.append(done_item)
            events += self._sse(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": self._item_index, "item": done_item},
            )
            self._item_index += 1

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            self.stop_reason = delta.get("stop_reason")
            usage = data.get("usage", {})
            self.output_tokens = usage.get("output_tokens", self.output_tokens)
            if usage.get("cache_creation_input_tokens") is not None:
                self.cache_creation_input_tokens = usage["cache_creation_input_tokens"]
            if usage.get("cache_read_input_tokens") is not None:
                self.cache_read_input_tokens = usage["cache_read_input_tokens"]

        elif event_type == "message_stop":
            events += self._emit_completed()

        return events

    def _emit_completed(self) -> str:
        if self._finished:
            return ""
        self._finished = True

        status = "incomplete" if self.stop_reason == "max_tokens" else "completed"

        usage: dict[str, Any] = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }
        if self.cache_creation_input_tokens:
            usage["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        if self.cache_read_input_tokens:
            usage["cache_read_input_tokens"] = self.cache_read_input_tokens

        response: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": self.output_items,
            "usage": usage,
        }
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}

        return self._sse(
            "response.completed",
            {"type": "response.completed", "response": response},
        )

    def flush(self) -> str:
        """Flush pending state. Call after stream ends."""
        if not self._finished and self._started:
            return self._emit_completed()
        return ""


# --- Streaming Translation (OpenAI Chat Completions SSE -> Responses SSE) ---


class OpenAIChatToResponsesStreamTranslator:
    """Stateful translator converting OpenAI streaming chunks to Responses SSE events.

    Handles the deferred finish pattern: OpenAI may send a usage-only chunk
    after the finish_reason chunk, so we defer emitting response.completed
    until we receive usage or the stream ends.
    """

    def __init__(self, model: str):
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        self.output_items: list[dict[str, Any]] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._started = False
        self._finished = False
        self._item_index = 0
        self._current_text_parts: list[str] = []
        self._has_text_item = False
        self._current_tool: dict[str, Any] | None = None
        self._current_tool_args: str = ""
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._finish_reason: str | None = None

    def _sse(self, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _emit_created(self) -> str:
        self._started = True
        return self._sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "created_at": self.created_at,
                    "status": "in_progress",
                    "model": self.model,
                    "output": [],
                    "usage": {
                        "input_tokens": self.input_tokens,
                        "output_tokens": 0,
                        "total_tokens": self.input_tokens,
                    },
                },
            },
        )

    def translate_chunk(self, chunk: dict[str, Any]) -> str:
        events = ""

        if not self._started:
            usage = chunk.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = usage["prompt_tokens"]
            events += self._emit_created()

        choices = chunk.get("choices") or []
        usage = chunk.get("usage") or {}

        # Handle usage-only chunk (final chunk with include_usage)
        if not choices and usage:
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = usage["prompt_tokens"]
            if usage.get("completion_tokens") is not None:
                self.output_tokens = usage["completion_tokens"]
            # If we already saw finish_reason, emit deferred completed now
            if self._finish_reason is not None and not self._finished:
                events += self._close_current_items()
                events += self._emit_completed()
            return events

        choice = choices[0] if choices else {}
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle text content
        text_content = delta.get("content")
        if text_content is not None:
            if not self._has_text_item:
                self._has_text_item = True
                self._current_text_parts = []
                item = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                }
                events += self._sse(
                    "response.output_item.added",
                    {"type": "response.output_item.added", "output_index": self._item_index, "item": item},
                )
            self._current_text_parts.append(text_content)
            events += self._sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "output_index": self._item_index,
                    "content_index": 0,
                    "delta": text_content,
                },
            )

        # Handle tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc in tool_calls:
            tc_index = tc.get("index", 0)
            func = tc.get("function", {})

            if tc_index not in self._tool_calls:
                # Close text item if open
                if self._has_text_item:
                    events += self._close_text_item()

                # Start new function_call
                tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:24]}")
                tc_name = func.get("name", "")
                self._tool_calls[tc_index] = {
                    "id": tc_id,
                    "name": tc_name,
                    "arguments": "",
                }
                self._current_tool = {
                    "type": "function_call",
                    "name": tc_name,
                    "arguments": "",
                    "call_id": tc_id,
                }
                self._current_tool_args = ""
                events += self._sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": self._item_index,
                        "item": self._current_tool,
                    },
                )

            # Accumulate arguments
            if func.get("arguments"):
                self._tool_calls[tc_index]["arguments"] += func["arguments"]
                self._current_tool_args += func["arguments"]
                events += self._sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": self._item_index,
                        "delta": func["arguments"],
                    },
                )

        # Handle finish_reason
        if finish_reason is not None:
            self._finish_reason = finish_reason

            # Capture usage from this chunk if present
            if usage.get("completion_tokens") is not None:
                self.output_tokens = usage["completion_tokens"]
            if usage.get("prompt_tokens") is not None:
                self.input_tokens = usage["prompt_tokens"]

            # If usage provided in this chunk, emit completed immediately
            if usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None:
                events += self._close_current_items()
                events += self._emit_completed()

        return events

    def _close_text_item(self) -> str:
        if not self._has_text_item:
            return ""
        self._has_text_item = False
        done_item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(self._current_text_parts)}],
        }
        self.output_items.append(done_item)
        events = self._sse(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": self._item_index, "item": done_item},
        )
        self._item_index += 1
        self._current_text_parts = []
        return events

    def _close_tool_item(self) -> str:
        if self._current_tool is None:
            return ""
        done_item = {**self._current_tool, "arguments": self._current_tool_args}
        self.output_items.append(done_item)
        events = self._sse(
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": self._item_index, "item": done_item},
        )
        self._item_index += 1
        self._current_tool = None
        self._current_tool_args = ""
        return events

    def _close_current_items(self) -> str:
        events = ""
        if self._has_text_item:
            events += self._close_text_item()
        if self._current_tool is not None:
            events += self._close_tool_item()
        return events

    def _emit_completed(self) -> str:
        if self._finished:
            return ""
        self._finished = True

        status = "incomplete" if self._finish_reason == "length" else "completed"

        response: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": self.output_items,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        }
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}

        return self._sse(
            "response.completed",
            {"type": "response.completed", "response": response},
        )

    def flush(self) -> str:
        """Flush pending state. Call after stream ends."""
        events = ""
        events += self._close_current_items()
        if self._finish_reason is not None and not self._finished:
            events += self._emit_completed()
        return events
