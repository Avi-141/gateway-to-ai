"""Tests for claudegate/copilot_translate.py."""

import json

from claudegate.copilot_translate import (
    StreamTranslator,
    _translate_content_to_openai,
    _translate_finish_reason,
    _translate_tool_choice,
    _translate_tools,
    anthropic_to_openai_request,
    estimate_input_tokens,
    has_server_tools,
    openai_to_anthropic_response,
    strip_server_tools,
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


# --- has_server_tools ---


class TestHasServerTools:
    def test_no_tools(self):
        assert has_server_tools({}) is False
        assert has_server_tools({"tools": []}) is False

    def test_only_custom_tools(self):
        body = {
            "tools": [
                {"name": "get_weather", "description": "d", "input_schema": {}},
                {"type": "custom", "name": "fn2", "description": "d", "input_schema": {}},
            ]
        }
        assert has_server_tools(body) is False

    def test_web_search_tool(self):
        body = {
            "tools": [
                {"type": "web_search_20250305", "name": "web_search"},
            ]
        }
        assert has_server_tools(body) is True

    def test_mixed_tools(self):
        body = {
            "tools": [
                {"name": "get_weather", "description": "d", "input_schema": {}},
                {"type": "web_search_20250305", "name": "web_search"},
            ]
        }
        assert has_server_tools(body) is True


# --- strip_server_tools ---


class TestStripServerTools:
    def test_removes_server_tools_keeps_custom(self):
        body = {
            "tools": [
                {"name": "get_weather", "description": "d", "input_schema": {}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = strip_server_tools(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert "tool_choice" in result

    def test_removes_tools_and_tool_choice_when_all_server(self):
        body = {
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = strip_server_tools(body)
        assert "tools" not in result
        assert "tool_choice" not in result

    def test_strips_server_tool_use_from_assistant_messages(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search"},
                        {"type": "server_tool_use", "id": "st_1", "name": "web_search"},
                    ],
                },
            ]
        }
        result = strip_server_tools(body)
        assert len(result["messages"]) == 1
        assert len(result["messages"][0]["content"]) == 1
        assert result["messages"][0]["content"][0]["type"] == "text"

    def test_strips_web_search_tool_result_from_user_messages(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "web_search_tool_result", "tool_use_id": "st_1", "content": []},
                        {"type": "text", "text": "What did you find?"},
                    ],
                },
            ]
        }
        result = strip_server_tools(body)
        assert len(result["messages"]) == 1
        assert len(result["messages"][0]["content"]) == 1
        assert result["messages"][0]["content"][0]["type"] == "text"

    def test_drops_empty_messages_after_stripping(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "web_search_tool_result", "tool_use_id": "st_1", "content": []},
                    ],
                },
                {"role": "user", "content": "follow up"},
            ]
        }
        result = strip_server_tools(body)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "follow up"

    def test_does_not_mutate_original(self):
        body = {
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "auto"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        strip_server_tools(body)
        # Original should be untouched
        assert len(body["tools"]) == 1
        assert "tool_choice" in body

    def test_string_content_messages_unaffected(self):
        body = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ]
        }
        result = strip_server_tools(body)
        assert len(result["messages"]) == 2


# --- _translate_tools with server tools ---


class TestTranslateToolsServerToolFiltering:
    def test_skips_server_tools(self):
        tools = [
            {"name": "fn", "description": "does stuff", "input_schema": {"type": "object"}},
            {"type": "web_search_20250305", "name": "web_search"},
        ]
        result = _translate_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "fn"

    def test_all_server_tools_returns_empty(self):
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        result = _translate_tools(tools)
        assert result == []


# --- anthropic_to_openai_request with server tools ---


class TestAnthropicToOpenAIRequestServerTools:
    def test_server_tools_filtered_no_tools_or_tool_choice_in_output(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "search for cats"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "tool_choice": {"type": "auto"},
        }
        result = anthropic_to_openai_request(body, "m")
        assert "tools" not in result
        assert "tool_choice" not in result

    def test_mixed_tools_keeps_custom_in_output(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "search for cats"}],
            "tools": [
                {"name": "fn", "description": "d", "input_schema": {}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
            "tool_choice": {"type": "auto"},
        }
        result = anthropic_to_openai_request(body, "m")
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "fn"
        assert result["tool_choice"] == "auto"

    def test_server_tool_use_blocks_dropped_from_assistant(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "result"},
                        {"type": "server_tool_use", "id": "st_1", "name": "web_search"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        msg = result["messages"][0]
        assert msg["content"] == "result"
        assert "tool_calls" not in msg

    def test_web_search_tool_result_dropped_from_user(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "web_search_tool_result", "tool_use_id": "st_1", "content": []},
                        {"type": "text", "text": "What did you find?"},
                    ],
                },
            ],
        }
        result = anthropic_to_openai_request(body, "m")
        # Only the text part should remain as a user message
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


# --- estimate_input_tokens ---


class TestEstimateInputTokens:
    def test_string_content_message(self):
        body = {"messages": [{"role": "user", "content": "Hello, world!"}]}
        result = estimate_input_tokens(body)
        assert result > 0

    def test_list_content_text_blocks(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ]
        }
        result = estimate_input_tokens(body)
        assert result > 0

    def test_tool_use_block(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "NYC"}},
                    ],
                }
            ]
        }
        result = estimate_input_tokens(body)
        assert result > 0

    def test_tool_result_string(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "72°F sunny"},
                    ],
                }
            ]
        }
        result = estimate_input_tokens(body)
        assert result > 0

    def test_tool_result_list_content(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "result data"}],
                        },
                    ],
                }
            ]
        }
        result = estimate_input_tokens(body)
        assert result > 0

    def test_system_prompt_string(self):
        body = {"system": "You are a helpful assistant.", "messages": []}
        result = estimate_input_tokens(body)
        assert result > 0

    def test_system_prompt_blocks(self):
        body = {
            "system": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            "messages": [],
        }
        result = estimate_input_tokens(body)
        assert result > 0

    def test_tools_definitions(self):
        body = {
            "messages": [],
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
        }
        result = estimate_input_tokens(body)
        assert result > 0

    def test_empty_body(self):
        assert estimate_input_tokens({}) == 0

    def test_empty_string_content(self):
        body = {"messages": [{"role": "user", "content": ""}]}
        assert estimate_input_tokens(body) == 0

    def test_no_double_counting_text_block(self):
        """Text blocks should only be counted once (via block_type check, not generic text extraction)."""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "hello"},
                    ],
                }
            ]
        }
        # Count for just the text
        text_only = estimate_input_tokens({"messages": [{"role": "user", "content": "hello"}]})
        result = estimate_input_tokens(body)
        # Should be roughly the same (both count "hello" once)
        assert result == text_only


# --- Additional StreamTranslator tests ---


class TestStreamTranslatorUsage:
    def test_usage_only_chunk_emits_deferred_message_end(self, openai_streaming_chunks_with_usage):
        t = StreamTranslator("model", estimated_input_tokens=100)
        all_events = ""
        for chunk in openai_streaming_chunks_with_usage:
            all_events += t.translate_chunk(chunk)

        assert "message_start" in all_events
        assert "message_delta" in all_events
        assert "message_stop" in all_events
        # Usage-only chunk should update the real prompt_tokens internally
        assert t.input_tokens == 42
        assert t.output_tokens == 3

    def test_flush_fallback_when_no_usage_chunk(self):
        t = StreamTranslator("model", estimated_input_tokens=50)
        # Start
        t.translate_chunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        # Finish with no usage
        t.translate_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        # No message_stop yet (waiting for usage chunk)
        assert not t._finished

        # Flush should emit message_delta/stop
        events = t.flush()
        assert "message_delta" in events
        assert "message_stop" in events
        assert t._finished

    def test_finished_guard_prevents_double_emit(self):
        t = StreamTranslator("model", estimated_input_tokens=10)
        t.translate_chunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        t.translate_chunk(
            {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        )
        assert t._finished
        # Second flush should be a no-op
        assert t.flush() == ""

    def test_estimated_input_tokens_in_message_start(self):
        t = StreamTranslator("model", estimated_input_tokens=999)
        events = t.translate_chunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        assert '"input_tokens": 999' in events

    def test_usage_chunk_overrides_estimated_tokens(self, openai_streaming_chunks_with_usage):
        t = StreamTranslator("model", estimated_input_tokens=999)
        all_events = ""
        for chunk in openai_streaming_chunks_with_usage:
            all_events += t.translate_chunk(chunk)
        # message_start has estimate (999), but message_delta should reflect real usage
        assert t.input_tokens == 42

    def test_cache_tokens_in_message_start(self):
        t = StreamTranslator("model")
        events = t.translate_chunk(
            {
                "choices": [{"delta": {"content": "hi"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 10, "cache_creation_input_tokens": 5, "cache_read_input_tokens": 3},
            }
        )
        assert '"cache_creation_input_tokens": 5' in events
        assert '"cache_read_input_tokens": 3' in events

    def test_cache_tokens_in_usage_only_chunk(self):
        t = StreamTranslator("model", estimated_input_tokens=10)
        t.translate_chunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        t.translate_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        events = t.translate_chunk(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cache_creation_input_tokens": 7,
                    "cache_read_input_tokens": 4,
                },
            }
        )
        assert "message_delta" in events
        assert '"cache_creation_input_tokens": 7' in events
        assert '"cache_read_input_tokens": 4' in events

    def test_complete_sequence_with_usage_only_chunk(self, openai_streaming_chunks_with_usage):
        t = StreamTranslator("model", estimated_input_tokens=100)
        all_events = ""
        for chunk in openai_streaming_chunks_with_usage:
            all_events += t.translate_chunk(chunk)

        # Full sequence
        assert "message_start" in all_events
        assert "content_block_start" in all_events
        assert "text_delta" in all_events
        assert "Hello" in all_events
        assert " world" in all_events
        assert "content_block_stop" in all_events
        assert "message_delta" in all_events
        assert "message_stop" in all_events
        assert "end_turn" in all_events

    def test_zero_prompt_tokens_not_ignored(self):
        """Verify that prompt_tokens: 0 is not treated as falsy/missing."""
        t = StreamTranslator("model", estimated_input_tokens=100)
        t.translate_chunk({"choices": [{"delta": {"content": "hi"}, "finish_reason": None}]})
        t.translate_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        events = t.translate_chunk({"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 1}})
        assert "message_delta" in events
        assert t.input_tokens == 0


# --- Additional openai_to_anthropic_response tests ---


class TestOpenAIToAnthropicResponseCacheFields:
    def test_cache_token_fields(self):
        resp = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 7,
            },
        }
        result = openai_to_anthropic_response(resp, "model")
        assert result["usage"]["cache_creation_input_tokens"] == 3
        assert result["usage"]["cache_read_input_tokens"] == 7

    def test_cache_token_fields_default_to_zero(self):
        resp = {
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = openai_to_anthropic_response(resp, "model")
        assert result["usage"]["cache_creation_input_tokens"] == 0
        assert result["usage"]["cache_read_input_tokens"] == 0
