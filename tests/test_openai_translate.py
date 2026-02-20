"""Tests for claudegate/openai_translate.py."""

import json

from claudegate.openai_translate import (
    ReverseStreamTranslator,
    _translate_stop_reason,
    _translate_tool_choice,
    _translate_tools,
    anthropic_to_openai_response,
    openai_to_anthropic_request,
    parse_anthropic_sse,
)

# --- openai_to_anthropic_request ---


class TestOpenAIToAnthropicRequest:
    def test_basic_request(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = openai_to_anthropic_request(body)
        assert result["model"] == "claude-sonnet-4-5-20250929"
        assert result["max_tokens"] == 4096  # default
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_system_extraction(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_to_anthropic_request(body)
        assert result["system"] == "You are helpful"
        # System message should not appear in messages
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_multiple_system_messages(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "system", "content": "Rule 1"},
                {"role": "system", "content": "Rule 2"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_to_anthropic_request(body)
        assert result["system"] == "Rule 1\nRule 2"

    def test_max_tokens_passthrough(self):
        body = {
            "model": "x",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = openai_to_anthropic_request(body)
        assert result["max_tokens"] == 512

    def test_max_tokens_default_when_absent(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = openai_to_anthropic_request(body)
        assert result["max_tokens"] == 4096

    def test_tools_translation(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ],
        }
        result = openai_to_anthropic_request(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["description"] == "Get weather"
        assert result["tools"][0]["input_schema"]["type"] == "object"

    def test_tool_choice_auto(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "fn", "parameters": {}}}],
            "tool_choice": "auto",
        }
        result = openai_to_anthropic_request(body)
        assert result["tool_choice"] == {"type": "auto"}

    def test_tool_choice_required(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "fn", "parameters": {}}}],
            "tool_choice": "required",
        }
        result = openai_to_anthropic_request(body)
        assert result["tool_choice"] == {"type": "any"}

    def test_tool_choice_none(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "fn", "parameters": {}}}],
            "tool_choice": "none",
        }
        result = openai_to_anthropic_request(body)
        assert result["tool_choice"] == {"type": "none"}

    def test_tool_choice_specific_function(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }
        result = openai_to_anthropic_request(body)
        assert result["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_assistant_with_tool_calls(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                        }
                    ],
                },
            ],
        }
        result = openai_to_anthropic_request(body)
        msg = result["messages"][1]
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Let me check"
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["name"] == "get_weather"
        assert msg["content"][1]["input"] == {"city": "NYC"}

    def test_tool_messages(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "tool", "tool_call_id": "call_1", "content": "72F and sunny"},
            ],
        }
        result = openai_to_anthropic_request(body)
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "call_1"
        assert msg["content"][0]["content"] == "72F and sunny"

    def test_consecutive_tool_messages_merged(self):
        body = {
            "model": "x",
            "messages": [
                {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
                {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
            ],
        }
        result = openai_to_anthropic_request(body)
        # Should be merged into a single user message with two tool_result blocks
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["tool_use_id"] == "call_1"
        assert msg["content"][1]["tool_use_id"] == "call_2"

    def test_temperature_top_p_stop(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["END", "STOP"],
        }
        result = openai_to_anthropic_request(body)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stop_sequences"] == ["END", "STOP"]

    def test_stream_flag(self):
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        }
        result = openai_to_anthropic_request(body)
        assert result["stream"] is True

    def test_user_image_data_url(self):
        """Data URL image_url parts are converted to Anthropic base64 image blocks."""
        body = {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBOR..."},
                        },
                    ],
                }
            ],
        }
        result = openai_to_anthropic_request(body)
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "What is this?"}
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"]["type"] == "base64"
        assert msg["content"][1]["source"]["media_type"] == "image/png"
        assert msg["content"][1]["source"]["data"] == "iVBOR..."

    def test_user_image_http_url(self):
        """HTTP URL image_url parts are converted to Anthropic url image blocks."""
        body = {
            "model": "x",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/img.png"},
                        },
                    ],
                }
            ],
        }
        result = openai_to_anthropic_request(body)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][0]["source"]["type"] == "url"
        assert msg["content"][0]["source"]["url"] == "https://example.com/img.png"

    def test_user_string_content_unchanged(self):
        """String content in user messages passes through unchanged."""
        body = {
            "model": "x",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = openai_to_anthropic_request(body)
        assert result["messages"][0]["content"] == "Hello"


# --- _translate_tools ---


class TestTranslateTools:
    def test_single_tool(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "fn",
                    "description": "does stuff",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = _translate_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "fn"
        assert result[0]["description"] == "does stuff"
        assert result[0]["input_schema"] == {"type": "object"}

    def test_empty(self):
        assert _translate_tools([]) == []


# --- _translate_tool_choice ---


class TestTranslateToolChoice:
    def test_auto_string(self):
        assert _translate_tool_choice("auto") == {"type": "auto"}

    def test_required_string(self):
        assert _translate_tool_choice("required") == {"type": "any"}

    def test_none_string(self):
        assert _translate_tool_choice("none") == {"type": "none"}

    def test_specific_function(self):
        result = _translate_tool_choice({"type": "function", "function": {"name": "fn"}})
        assert result == {"type": "tool", "name": "fn"}


# --- _translate_stop_reason ---


class TestTranslateStopReason:
    def test_end_turn(self):
        assert _translate_stop_reason("end_turn") == "stop"

    def test_max_tokens(self):
        assert _translate_stop_reason("max_tokens") == "length"

    def test_tool_use(self):
        assert _translate_stop_reason("tool_use") == "tool_calls"

    def test_stop_sequence(self):
        assert _translate_stop_reason("stop_sequence") == "stop"

    def test_none(self):
        assert _translate_stop_reason(None) == "stop"


# --- anthropic_to_openai_response ---


class TestAnthropicToOpenAIResponse:
    def test_text_response(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello there!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_to_openai_response(anthropic_resp, "claude-sonnet-4.5")
        assert result["object"] == "chat.completion"
        assert result["model"] == "claude-sonnet-4.5"
        assert result["choices"][0]["message"]["content"] == "Hello there!"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_tool_use_response(self):
        anthropic_resp = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = anthropic_to_openai_response(anthropic_resp, "model")
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        msg = result["choices"][0]["message"]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "toolu_123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "NYC"}

    def test_mixed_text_and_tool_use(self):
        anthropic_resp = {
            "content": [
                {"type": "text", "text": "I'll check"},
                {"type": "tool_use", "id": "t1", "name": "fn", "input": {}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        result = anthropic_to_openai_response(anthropic_resp, "model")
        msg = result["choices"][0]["message"]
        assert msg["content"] == "I'll check"
        assert len(msg["tool_calls"]) == 1

    def test_stop_reason_mapping(self):
        for anthropic_reason, openai_reason in [
            ("end_turn", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "tool_calls"),
        ]:
            resp = {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": anthropic_reason,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
            result = anthropic_to_openai_response(resp, "model")
            assert result["choices"][0]["finish_reason"] == openai_reason

    def test_usage_calculation(self):
        resp = {
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        result = anthropic_to_openai_response(resp, "model")
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 50
        assert result["usage"]["total_tokens"] == 150


# --- parse_anthropic_sse ---


class TestParseAnthropicSSE:
    def test_single_event(self):
        raw = 'event: message_start\ndata: {"type": "message_start"}\n\n'
        events = parse_anthropic_sse(raw)
        assert len(events) == 1
        assert events[0][0] == "message_start"
        assert events[0][1]["type"] == "message_start"

    def test_multiple_events(self):
        raw = (
            'event: content_block_start\ndata: {"type": "content_block_start"}\n\n'
            'event: content_block_delta\ndata: {"type": "content_block_delta"}\n\n'
        )
        events = parse_anthropic_sse(raw)
        assert len(events) == 2

    def test_done_event_skipped(self):
        raw = "event: done\ndata: [DONE]\n\n"
        events = parse_anthropic_sse(raw)
        assert len(events) == 0

    def test_invalid_json_skipped(self):
        raw = "event: foo\ndata: not-json\n\n"
        events = parse_anthropic_sse(raw)
        assert len(events) == 0


# --- ReverseStreamTranslator ---


class TestReverseStreamTranslator:
    def test_message_start_emits_role(self):
        t = ReverseStreamTranslator("model")
        result = t.translate_event(
            "message_start",
            {"type": "message_start", "message": {"role": "assistant"}},
        )
        assert "chat.completion.chunk" in result
        assert '"role": "assistant"' in result

    def test_text_delta(self):
        t = ReverseStreamTranslator("model")
        t.translate_event("message_start", {"type": "message_start", "message": {}})
        result = t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
        )
        assert '"content": "Hello"' in result

    def test_tool_call_start(self):
        t = ReverseStreamTranslator("model")
        t.translate_event("message_start", {"type": "message_start", "message": {}})
        result = t.translate_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "id": "t1", "name": "fn"},
            },
        )
        assert "tool_calls" in result
        assert '"name": "fn"' in result

    def test_tool_call_arguments(self):
        t = ReverseStreamTranslator("model")
        t.translate_event("message_start", {"type": "message_start", "message": {}})
        result = t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "delta": {"type": "input_json_delta", "partial_json": '{"a":'}},
        )
        assert "tool_calls" in result
        # The partial_json is embedded inside a JSON string, so quotes are escaped
        parsed = json.loads(result.split("data: ")[1].split("\n")[0])
        assert parsed["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] == '{"a":'

    def test_message_delta_finish_reason(self):
        t = ReverseStreamTranslator("model")
        t.translate_event("message_start", {"type": "message_start", "message": {}})
        result = t.translate_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        )
        assert '"finish_reason": "stop"' in result

    def test_message_stop_emits_done(self):
        t = ReverseStreamTranslator("model")
        result = t.translate_event("message_stop", {"type": "message_stop"})
        assert "data: [DONE]" in result

    def test_complete_text_sequence(self):
        t = ReverseStreamTranslator("model")
        all_output = ""

        all_output += t.translate_event(
            "message_start",
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
        )
        all_output += t.translate_event(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        )
        all_output += t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        )
        all_output += t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        )
        all_output += t.translate_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        all_output += t.translate_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        )
        all_output += t.translate_event("message_stop", {"type": "message_stop"})

        assert '"role": "assistant"' in all_output
        assert "Hello" in all_output
        assert " world" in all_output
        assert '"finish_reason": "stop"' in all_output
        assert "data: [DONE]" in all_output

    def test_complete_tool_call_sequence(self):
        t = ReverseStreamTranslator("model")
        all_output = ""

        all_output += t.translate_event(
            "message_start",
            {"type": "message_start", "message": {}},
        )
        all_output += t.translate_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {}},
            },
        )
        all_output += t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"ci'}},
        )
        all_output += t.translate_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": 'ty":"NYC"}'},
            },
        )
        all_output += t.translate_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        all_output += t.translate_event(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        )
        all_output += t.translate_event("message_stop", {"type": "message_stop"})

        assert "get_weather" in all_output
        assert '"finish_reason": "tool_calls"' in all_output
        assert "data: [DONE]" in all_output

    def test_multiple_tool_calls_sequential_indices(self):
        t = ReverseStreamTranslator("model")
        all_output = ""

        all_output += t.translate_event("message_start", {"type": "message_start", "message": {}})
        # First tool call
        all_output += t.translate_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "t1", "name": "fn1", "input": {}},
            },
        )
        all_output += t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        )
        all_output += t.translate_event("content_block_stop", {"type": "content_block_stop", "index": 0})
        # Second tool call
        all_output += t.translate_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "t2", "name": "fn2", "input": {}},
            },
        )
        all_output += t.translate_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        )
        all_output += t.translate_event("content_block_stop", {"type": "content_block_stop", "index": 1})
        all_output += t.translate_event(
            "message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}
        )
        all_output += t.translate_event("message_stop", {"type": "message_stop"})

        # Parse all SSE chunks and collect tool call indices
        indices = []
        for line in all_output.split("\n"):
            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                if "tool_calls" in delta:
                    indices.append(delta["tool_calls"][0]["index"])

        # First tool call chunks should have index 0, second should have index 1
        assert 0 in indices
        assert 1 in indices

    def test_translate_sse_raw(self):
        t = ReverseStreamTranslator("model")
        raw = (
            'event: message_start\ndata: {"type": "message_start", "message": {"role": "assistant"}}\n\n'
            "event: content_block_delta\n"
            'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}\n\n'
            'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}\n\n'
            'event: message_stop\ndata: {"type": "message_stop"}\n\n'
        )
        result = t.translate_sse(raw)
        assert "Hi" in result
        assert "data: [DONE]" in result
