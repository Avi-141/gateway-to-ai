"""Tests for claudegate/responses_translate.py."""

import json

from claudegate.responses_translate import (
    ResponsesStreamTranslator,
    ResponsesToOpenAIStreamTranslator,
    anthropic_to_responses_request,
    openai_chat_to_responses_request,
    responses_to_anthropic_response,
    responses_to_openai_chat_response,
)

# --- anthropic_to_responses_request ---


class TestAnthropicToResponsesRequest:
    def test_simple_text_message(self):
        body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["model"] == "gpt-5.2-codex"
        assert result["max_output_tokens"] == 1024
        assert len(result["input"]) == 1
        assert result["input"][0]["role"] == "user"
        assert result["input"][0]["content"] == [{"type": "input_text", "text": "Hello"}]

    def test_system_prompt_string(self):
        body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["instructions"] == "You are helpful."

    def test_system_prompt_blocks(self):
        body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["instructions"] == "Part 1\nPart 2"

    def test_assistant_text_message(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert len(result["input"]) == 3
        assert result["input"][1]["role"] == "assistant"
        assert result["input"][1]["content"] == [{"type": "output_text", "text": "Hello there"}]

    def test_tool_use_and_result(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "call_abc123",
                            "name": "get_weather",
                            "input": {"location": "NYC"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_abc123",
                            "content": "72°F, sunny",
                        }
                    ],
                },
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        input_items = result["input"]

        # User message
        assert input_items[0]["role"] == "user"

        # Assistant text then function_call
        assert input_items[1]["role"] == "assistant"
        assert input_items[1]["content"] == [{"type": "output_text", "text": "Let me check."}]

        assert input_items[2]["type"] == "function_call"
        assert input_items[2]["name"] == "get_weather"
        assert input_items[2]["call_id"] == "call_abc123"
        assert json.loads(input_items[2]["arguments"]) == {"location": "NYC"}

        # Tool result
        assert input_items[3]["type"] == "function_call_output"
        assert input_items[3]["call_id"] == "call_abc123"
        assert input_items[3]["output"] == "72°F, sunny"

    def test_tool_definitions(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {"type": "object", "properties": {"loc": {"type": "string"}}},
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["parameters"] == {"type": "object", "properties": {"loc": {"type": "string"}}}

    def test_tool_choice_auto(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["tool_choice"] == "required"

    def test_temperature_and_top_p(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9

    def test_user_content_blocks(self):
        """User messages with content block arrays."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert len(result["input"]) == 2
        assert result["input"][0]["content"][0]["text"] == "Hello"
        assert result["input"][1]["content"][0]["text"] == "World"

    def test_tool_result_with_content_blocks(self):
        """Tool results with list content."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": [{"type": "text", "text": "result data"}],
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert result["input"][0]["type"] == "function_call_output"
        assert result["input"][0]["output"] == "result data"


# --- openai_chat_to_responses_request ---


class TestOpenAIChatToResponsesRequest:
    def test_simple_message(self):
        body = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = openai_chat_to_responses_request(body, "gpt-5.2-codex")
        assert result["model"] == "gpt-5.2-codex"
        assert len(result["input"]) == 1
        assert result["input"][0]["content"][0]["text"] == "Hello"

    def test_system_message(self):
        body = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
        }
        result = openai_chat_to_responses_request(body, "gpt-5.2-codex")
        assert result["instructions"] == "Be helpful"
        assert len(result["input"]) == 1

    def test_tool_calls_and_results(self):
        body = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "Checking",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "72°F"},
            ],
        }
        result = openai_chat_to_responses_request(body, "gpt-5.2-codex")
        input_items = result["input"]

        assert input_items[0]["role"] == "user"
        assert input_items[1]["role"] == "assistant"
        assert input_items[2]["type"] == "function_call"
        assert input_items[2]["name"] == "get_weather"
        assert input_items[2]["call_id"] == "call_abc"
        assert input_items[3]["type"] == "function_call_output"
        assert input_items[3]["call_id"] == "call_abc"
        assert input_items[3]["output"] == "72°F"

    def test_tools_translation(self):
        body = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search the web",
                        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    },
                }
            ],
        }
        result = openai_chat_to_responses_request(body, "gpt-5.2-codex")
        assert result["tools"][0]["name"] == "search"
        assert result["tools"][0]["parameters"]["properties"]["q"]["type"] == "string"

    def test_max_tokens(self):
        body = {
            "model": "gpt-5.2-codex",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 2048,
        }
        result = openai_chat_to_responses_request(body, "gpt-5.2-codex")
        assert result["max_output_tokens"] == 2048


# --- responses_to_anthropic_response ---


class TestResponsesToAnthropicResponse:
    def test_text_response(self):
        resp = {
            "id": "resp_abc",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert result["id"] == "resp_abc"
        assert result["model"] == "claude-sonnet-4-5"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_call_response(self):
        resp = {
            "id": "resp_abc",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location":"NYC"}',
                    "call_id": "call_xyz",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["id"] == "call_xyz"
        assert result["content"][0]["name"] == "get_weather"
        assert result["content"][0]["input"] == {"location": "NYC"}
        assert result["stop_reason"] == "tool_use"

    def test_mixed_text_and_tool_call(self):
        resp = {
            "id": "resp_abc",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Let me check."}]},
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": '{"q":"test"}',
                    "call_id": "call_1",
                },
            ],
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"
        assert result["stop_reason"] == "tool_use"

    def test_reasoning_skipped(self):
        resp = {
            "id": "resp_abc",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [{"type": "output_text", "text": "Done."}]},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert len(result["content"]) == 1
        assert result["content"][0]["text"] == "Done."

    def test_empty_output(self):
        resp = {"id": "resp_abc", "output": [], "usage": {}}
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_incomplete_max_tokens(self):
        resp = {
            "id": "resp_abc",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Partial..."}]}],
            "usage": {"input_tokens": 10, "output_tokens": 4096},
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert result["stop_reason"] == "max_tokens"

    def test_cache_token_fields(self):
        resp = {
            "id": "resp_abc",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 7,
            },
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert result["usage"]["cache_creation_input_tokens"] == 3
        assert result["usage"]["cache_read_input_tokens"] == 7

    def test_cache_token_fields_default_to_zero(self):
        resp = {
            "id": "resp_abc",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert result["usage"]["cache_creation_input_tokens"] == 0
        assert result["usage"]["cache_read_input_tokens"] == 0


# --- responses_to_openai_chat_response ---


class TestResponsesToOpenAIChatResponse:
    def test_text_response(self):
        resp = {
            "id": "resp_abc",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello!"}]}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = responses_to_openai_chat_response(resp, "gpt-5.2-codex")
        assert result["object"] == "chat.completion"
        assert result["model"] == "gpt-5.2-codex"
        choice = result["choices"][0]
        assert choice["message"]["content"] == "Hello!"
        assert choice["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5

    def test_tool_call_response(self):
        resp = {
            "id": "resp_abc",
            "output": [
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": '{"q":"test"}',
                    "call_id": "call_1",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = responses_to_openai_chat_response(resp, "gpt-5.2-codex")
        choice = result["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tc = choice["message"]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "search"
        assert tc["function"]["arguments"] == '{"q":"test"}'

    def test_reasoning_skipped(self):
        resp = {
            "id": "resp_abc",
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [{"type": "output_text", "text": "Done."}]},
            ],
            "usage": {},
        }
        result = responses_to_openai_chat_response(resp, "gpt-5.2-codex")
        assert result["choices"][0]["message"]["content"] == "Done."


# --- ResponsesStreamTranslator ---


class TestResponsesStreamTranslator:
    def test_text_stream_sequence(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")

        events = translator.translate_event("response.created", {"response": {"usage": {"input_tokens": 10}}})
        assert "message_start" in events
        assert '"input_tokens": 10' in events

        events = translator.translate_event("response.output_text.delta", {"delta": "Hello"})
        assert "content_block_start" in events  # first text block
        assert "text_delta" in events
        assert "Hello" in events

        events = translator.translate_event("response.output_text.delta", {"delta": " world"})
        assert "text_delta" in events
        assert " world" in events
        assert "content_block_start" not in events  # no new block

        events = translator.translate_event("response.output_item.done", {})
        assert "content_block_stop" in events

        events = translator.translate_event(
            "response.completed",
            {
                "response": {
                    "status": "completed",
                    "usage": {"output_tokens": 5},
                    "output": [],
                }
            },
        )
        assert "message_delta" in events
        assert "end_turn" in events
        assert "message_stop" in events

    def test_tool_call_stream_sequence(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")

        translator.translate_event("response.created", {"response": {}})

        events = translator.translate_event(
            "response.output_item.added",
            {"item": {"type": "function_call", "call_id": "call_abc", "name": "get_weather"}},
        )
        assert "content_block_start" in events
        assert "tool_use" in events
        assert "call_abc" in events
        assert "get_weather" in events

        events = translator.translate_event("response.function_call_arguments.delta", {"delta": '{"loc'})
        assert "input_json_delta" in events
        assert "loc" in events

        events = translator.translate_event("response.function_call_arguments.delta", {"delta": 'ation":"NYC"}'})
        assert "input_json_delta" in events

        events = translator.translate_event("response.output_item.done", {})
        assert "content_block_stop" in events

        events = translator.translate_event(
            "response.completed",
            {
                "response": {
                    "status": "completed",
                    "usage": {"output_tokens": 20},
                    "output": [{"type": "function_call"}],
                }
            },
        )
        assert "tool_use" in events  # stop_reason
        assert "message_stop" in events

    def test_text_then_tool_call(self):
        """Text block should be closed before tool_use block starts."""
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")
        translator.translate_event("response.created", {"response": {}})

        translator.translate_event("response.output_text.delta", {"delta": "Checking..."})
        assert translator.has_text_block is True
        assert translator.current_block_type == "text"

        events = translator.translate_event(
            "response.output_item.added", {"item": {"type": "function_call", "call_id": "call_1", "name": "search"}}
        )
        # Should contain block_stop for text, then block_start for tool_use
        assert "content_block_stop" in events
        assert "content_block_start" in events
        assert "tool_use" in events

    def test_message_start_emitted_once(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")
        events1 = translator.translate_event("response.created", {"response": {}})
        assert "message_start" in events1

        events2 = translator.translate_event("response.output_text.delta", {"delta": "Hi"})
        assert "message_start" not in events2

    def test_cache_tokens_in_message_start(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")
        events = translator.translate_event(
            "response.created",
            {
                "response": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 3,
                    }
                }
            },
        )
        assert '"cache_creation_input_tokens": 5' in events
        assert '"cache_read_input_tokens": 3' in events

    def test_cache_tokens_in_message_delta(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")
        translator.translate_event("response.created", {"response": {}})
        translator.translate_event("response.output_text.delta", {"delta": "Hi"})
        events = translator.translate_event(
            "response.completed",
            {
                "response": {
                    "status": "completed",
                    "usage": {
                        "output_tokens": 5,
                        "cache_creation_input_tokens": 7,
                        "cache_read_input_tokens": 4,
                    },
                    "output": [],
                }
            },
        )
        assert '"cache_creation_input_tokens": 7' in events
        assert '"cache_read_input_tokens": 4' in events

    def test_estimated_input_tokens(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5", estimated_input_tokens=999)
        events = translator.translate_event("response.created", {"response": {}})
        assert '"input_tokens": 999' in events

    def test_flush_closes_open_block(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")
        translator.translate_event("response.created", {"response": {}})
        translator.translate_event("response.output_text.delta", {"delta": "Hi"})
        assert translator.current_block_type == "text"

        events = translator.flush()
        assert "content_block_stop" in events
        assert translator.current_block_type is None

    def test_flush_noop_when_no_open_block(self):
        translator = ResponsesStreamTranslator("claude-sonnet-4-5")
        translator.translate_event("response.created", {"response": {}})
        assert translator.flush() == ""


# --- ResponsesToOpenAIStreamTranslator ---


class TestResponsesToOpenAIStreamTranslator:
    def test_text_stream_sequence(self):
        translator = ResponsesToOpenAIStreamTranslator("gpt-5.2-codex")

        events = translator.translate_event("response.created", {})
        assert '"role": "assistant"' in events

        events = translator.translate_event("response.output_text.delta", {"delta": "Hello"})
        assert '"content": "Hello"' in events

        events = translator.translate_event("response.completed", {"response": {"output": [], "usage": {}}})
        assert '"finish_reason": "stop"' in events
        assert "[DONE]" in events

    def test_tool_call_stream(self):
        translator = ResponsesToOpenAIStreamTranslator("gpt-5.2-codex")
        translator.translate_event("response.created", {})

        events = translator.translate_event(
            "response.output_item.added", {"item": {"type": "function_call", "call_id": "call_1", "name": "search"}}
        )
        assert "tool_calls" in events
        assert "search" in events

        events = translator.translate_event("response.function_call_arguments.delta", {"delta": '{"q":"test"}'})
        assert "tool_calls" in events
        assert "test" in events

        translator.translate_event("response.output_item.done", {"item": {"type": "function_call"}})
        assert translator.tool_call_index == 1

        events = translator.translate_event(
            "response.completed", {"response": {"output": [{"type": "function_call"}], "usage": {}}}
        )
        assert '"finish_reason": "tool_calls"' in events

    def test_incomplete_status_returns_length(self):
        translator = ResponsesToOpenAIStreamTranslator("gpt-5.2-codex")
        translator.translate_event("response.created", {})
        translator.translate_event("response.output_text.delta", {"delta": "partial"})

        events = translator.translate_event(
            "response.completed",
            {"response": {"status": "incomplete", "output": [], "usage": {}}},
        )
        assert '"finish_reason": "length"' in events
        assert "[DONE]" in events


# --- Round-trip tests ---


class TestToolCallRoundTrip:
    def test_anthropic_tool_call_id_preserved(self):
        """call_id is preserved through Anthropic -> Responses -> Anthropic."""
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": "72°F",
                        }
                    ],
                },
            ],
        }
        req = anthropic_to_responses_request(body, "gpt-5.2-codex")

        # Find function_call and function_call_output
        fc = next(i for i in req["input"] if i.get("type") == "function_call")
        fco = next(i for i in req["input"] if i.get("type") == "function_call_output")
        assert fc["call_id"] == "toolu_abc123"
        assert fco["call_id"] == "toolu_abc123"

        # Now simulate response coming back
        resp = {
            "id": "resp_1",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                    "call_id": "call_new_123",
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        anthropic_resp = responses_to_anthropic_response(resp, "claude-sonnet-4-5")
        assert anthropic_resp["content"][0]["id"] == "call_new_123"

    def test_openai_tool_call_id_preserved(self):
        """call_id is preserved through OpenAI -> Responses -> OpenAI."""
        body = {
            "model": "gpt-5.2-codex",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "72°F"},
            ],
        }
        req = openai_chat_to_responses_request(body, "gpt-5.2-codex")

        fc = next(i for i in req["input"] if i.get("type") == "function_call")
        fco = next(i for i in req["input"] if i.get("type") == "function_call_output")
        assert fc["call_id"] == "call_abc"
        assert fco["call_id"] == "call_abc"

        resp = {
            "id": "resp_1",
            "output": [
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                    "call_id": "call_new",
                }
            ],
            "usage": {},
        }
        openai_resp = responses_to_openai_chat_response(resp, "gpt-5.2-codex")
        assert openai_resp["choices"][0]["message"]["tool_calls"][0]["id"] == "call_new"
