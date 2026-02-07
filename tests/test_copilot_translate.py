"""Tests for claudegate/copilot_translate.py."""

import json

from claudegate.copilot_translate import (
    StreamTranslator,
    _translate_content_to_openai,
    _translate_finish_reason,
    _translate_tool_choice,
    _translate_tools,
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)

# --- _translate_content_to_openai ---


class TestTranslateContentToOpenAI:
    def test_string_passthrough(self):
        assert _translate_content_to_openai("hello") == "hello"

    def test_text_blocks_joined(self):
        blocks = [{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}]
        assert _translate_content_to_openai(blocks) == "foo\nbar"

    def test_image_block(self):
        blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png"}}]
        result = _translate_content_to_openai(blocks)
        assert "image/png" in result

    def test_mixed_blocks(self):
        blocks = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "id": "t1", "name": "fn", "input": {}},
        ]
        # tool_use blocks are skipped
        assert _translate_content_to_openai(blocks) == "hello"

    def test_string_blocks(self):
        blocks = ["hello", "world"]
        assert _translate_content_to_openai(blocks) == "hello\nworld"

    def test_empty_list(self):
        assert _translate_content_to_openai([]) == ""

    def test_none_input(self):
        assert _translate_content_to_openai(None) == ""


# --- _translate_tool_choice ---


class TestTranslateToolChoice:
    def test_auto(self):
        assert _translate_tool_choice({"type": "auto"}) == "auto"

    def test_any_becomes_required(self):
        assert _translate_tool_choice({"type": "any"}) == "required"

    def test_tool_becomes_function(self):
        result = _translate_tool_choice({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_none_type(self):
        assert _translate_tool_choice({"type": "none"}) == "none"

    def test_non_dict_passthrough(self):
        assert _translate_tool_choice("auto") == "auto"


# --- _translate_tools ---


class TestTranslateTools:
    def test_single_tool(self):
        tools = [{"name": "fn", "description": "does stuff", "input_schema": {"type": "object"}}]
        result = _translate_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "fn"
        assert result[0]["function"]["description"] == "does stuff"
        assert result[0]["function"]["parameters"] == {"type": "object"}

    def test_multiple_tools(self):
        tools = [
            {"name": "a", "description": "aa", "input_schema": {}},
            {"name": "b", "description": "bb", "input_schema": {}},
        ]
        assert len(_translate_tools(tools)) == 2

    def test_empty_list(self):
        assert _translate_tools([]) == []


# --- anthropic_to_openai_request ---


class TestAnthropicToOpenAIRequest:
    def test_basic_request(self):
        body = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai_request(body, "claude-sonnet-4.5")
        assert result["model"] == "claude-sonnet-4.5"
        assert result["max_tokens"] == 100
        # system message not present, just the user message
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_system_string(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are a bot",
        }
        result = anthropic_to_openai_request(body, "m")
        assert result["messages"][0] == {"role": "system", "content": "You are a bot"}

    def test_system_list(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "system": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}],
        }
        result = anthropic_to_openai_request(body, "m")
        assert result["messages"][0]["content"] == "part1\npart2"

    def test_assistant_with_text(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        assert result["messages"][1]["role"] == "assistant"
        assert result["messages"][1]["content"] == "hello"

    def test_assistant_with_tool_use_blocks(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        {"type": "tool_use", "id": "t1", "name": "fn", "input": {"a": 1}},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        msg = result["messages"][1]
        assert msg["content"] == "Let me check"
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "fn"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"a": 1}

    def test_assistant_with_thinking_blocks_dropped(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        msg = result["messages"][1]
        assert msg["content"] == "answer"
        assert "tool_calls" not in msg

    def test_user_with_tool_results(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "result text"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][0]["tool_call_id"] == "t1"
        assert result["messages"][0]["content"] == "result text"

    def test_user_with_mixed_content(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "res"},
                        {"type": "text", "text": "and also this"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        # tool message first, then user message with remaining content
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][1]["role"] == "user"

    def test_tools_and_tool_choice(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "fn", "description": "d", "input_schema": {}}],
            "tool_choice": {"type": "any"},
        }
        result = anthropic_to_openai_request(body, "m")
        assert len(result["tools"]) == 1
        assert result["tool_choice"] == "required"

    def test_temperature_top_p_stop(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "stop_sequences": ["END"],
        }
        result = anthropic_to_openai_request(body, "m")
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["stop"] == ["END"]

    def test_stream_flag(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        result = anthropic_to_openai_request(body, "m")
        assert result["stream"] is True

    def test_model_passthrough(self):
        body = {
            "model": "original",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = anthropic_to_openai_request(body, "override-model")
        assert result["model"] == "override-model"


# --- _translate_finish_reason ---


class TestTranslateFinishReason:
    def test_stop(self):
        assert _translate_finish_reason("stop") == "end_turn"

    def test_length(self):
        assert _translate_finish_reason("length") == "max_tokens"

    def test_tool_calls(self):
        assert _translate_finish_reason("tool_calls") == "tool_use"

    def test_content_filter(self):
        assert _translate_finish_reason("content_filter") == "end_turn"

    def test_none(self):
        assert _translate_finish_reason(None) == "end_turn"


# --- openai_to_anthropic_response ---


class TestOpenAIToAnthropicResponse:
    def test_basic_text(self, openai_chat_response):
        result = openai_to_anthropic_response(openai_chat_response, "claude-sonnet-4-5-20250929")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-sonnet-4-5-20250929"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello there!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        result = openai_to_anthropic_response(resp, "model")
        assert result["stop_reason"] == "tool_use"
        tool_block = result["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "NYC"}

    def test_invalid_json_arguments(self):
        resp = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "fn", "arguments": "not json"}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {},
        }
        result = openai_to_anthropic_response(resp, "model")
        assert result["content"][0]["input"] == {}

    def test_empty_content(self):
        resp = {
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = openai_to_anthropic_response(resp, "model")
        # Should have a fallback empty text block
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == ""

    def test_usage_mapping(self, openai_chat_response):
        result = openai_to_anthropic_response(openai_chat_response, "model")
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5


# --- StreamTranslator ---


class TestStreamTranslator:
    def test_first_chunk_emits_message_start(self):
        t = StreamTranslator("model")
        chunk = {"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]}
        events = t.translate_chunk(chunk)
        assert "event: message_start" in events
        assert '"type": "message_start"' in events

    def test_text_deltas(self):
        t = StreamTranslator("model")
        chunk1 = {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]}
        events = t.translate_chunk(chunk1)
        assert "text_delta" in events
        assert "Hello" in events

    def test_multiple_text_deltas(self):
        t = StreamTranslator("model")
        events1 = t.translate_chunk({"choices": [{"delta": {"content": "A"}, "finish_reason": None}]})
        events2 = t.translate_chunk({"choices": [{"delta": {"content": "B"}, "finish_reason": None}]})
        # First should have content_block_start, second should not
        assert "content_block_start" in events1
        assert "content_block_start" not in events2
        assert '"text": "B"' in events2

    def test_tool_call_start_and_arguments(self):
        t = StreamTranslator("model")
        # First an empty init chunk
        t.translate_chunk({"choices": [{"delta": {"content": ""}, "finish_reason": None}]})
        # Then tool call
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "fn", "arguments": '{"a":'}}]
                    },
                    "finish_reason": None,
                }
            ],
        }
        events = t.translate_chunk(chunk)
        assert "tool_use" in events
        assert "fn" in events
        assert "input_json_delta" in events

    def test_finish_events(self):
        t = StreamTranslator("model")
        # Start
        t.translate_chunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        # Finish
        events = t.translate_chunk(
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}
        )
        assert "content_block_stop" in events
        assert "message_delta" in events
        assert "message_stop" in events
        assert "end_turn" in events

    def test_complete_text_sequence(self, openai_streaming_chunks):
        t = StreamTranslator("model")
        all_events = ""
        for chunk in openai_streaming_chunks:
            all_events += t.translate_chunk(chunk)

        assert "message_start" in all_events
        assert "content_block_start" in all_events
        assert "text_delta" in all_events
        assert "Hello" in all_events
        assert " world" in all_events
        assert "content_block_stop" in all_events
        assert "message_delta" in all_events
        assert "message_stop" in all_events

    def test_complete_tool_sequence(self):
        t = StreamTranslator("model")
        chunks = [
            # Init
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            # Tool call start
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "fn", "arguments": ""}}]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            # Tool arguments
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"x":1}'}}]},
                        "finish_reason": None,
                    }
                ],
            },
            # Finish
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 10}},
        ]
        all_events = ""
        for c in chunks:
            all_events += t.translate_chunk(c)

        assert "message_start" in all_events
        assert "tool_use" in all_events
        assert "input_json_delta" in all_events
        assert "content_block_stop" in all_events
        assert "message_stop" in all_events

    def test_sse_format(self):
        t = StreamTranslator("model")
        events = t.translate_chunk({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]})
        # Each event should be "event: ...\ndata: ...\n\n"
        lines = events.strip().split("\n\n")
        for block in lines:
            parts = block.strip().split("\n")
            assert len(parts) >= 2
            assert parts[0].startswith("event: ")
            assert parts[1].startswith("data: ")
