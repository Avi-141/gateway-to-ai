"""Tests for claudegate/responses_translate.py."""

import json

from claudegate.responses_translate import (
    AnthropicToResponsesStreamTranslator,
    OpenAIChatToResponsesStreamTranslator,
    ResponsesStreamTranslator,
    ResponsesToOpenAIStreamTranslator,
    anthropic_to_responses_request,
    anthropic_to_responses_response,
    openai_chat_to_responses_request,
    openai_chat_to_responses_response,
    responses_to_anthropic_request,
    responses_to_anthropic_response,
    responses_to_openai_chat_request,
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


# --- responses_to_anthropic_request ---


class TestResponsesToAnthropicRequest:
    def test_simple_text(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        }
        result = responses_to_anthropic_request(body)
        assert result["model"] == "claude-sonnet-4-5"
        assert result["max_tokens"] == 4096  # default
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_instructions_to_system(self):
        body = {
            "model": "claude-sonnet-4-5",
            "instructions": "You are helpful.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        }
        result = responses_to_anthropic_request(body)
        assert result["system"] == "You are helpful."

    def test_max_output_tokens(self):
        body = {
            "model": "claude-sonnet-4-5",
            "max_output_tokens": 2048,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        }
        result = responses_to_anthropic_request(body)
        assert result["max_tokens"] == 2048

    def test_tools_function_type(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}},
                }
            ],
        }
        result = responses_to_anthropic_request(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tools"][0]["input_schema"] == {"type": "object", "properties": {"loc": {"type": "string"}}}

    def test_tool_choice_required(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "tool_choice": "required",
        }
        result = responses_to_anthropic_request(body)
        assert result["tool_choice"] == {"type": "any"}

    def test_tool_choice_function(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "tool_choice": {"type": "function", "name": "get_weather"},
        }
        result = responses_to_anthropic_request(body)
        assert result["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_function_call_and_output(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"location":"NYC"}',
                    "call_id": "call_abc",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": "72°F, sunny",
                },
            ],
        }
        result = responses_to_anthropic_request(body)
        messages = result["messages"]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Weather?"
        # function_call -> assistant with tool_use
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"][0]["type"] == "tool_use"
        assert messages[1]["content"][0]["name"] == "get_weather"
        assert messages[1]["content"][0]["id"] == "call_abc"
        # function_call_output -> user with tool_result
        assert messages[2]["role"] == "user"
        assert messages[2]["content"][0]["type"] == "tool_result"
        assert messages[2]["content"][0]["tool_use_id"] == "call_abc"

    def test_string_input(self):
        body = {"model": "claude-sonnet-4-5", "input": "Hello there"}
        result = responses_to_anthropic_request(body)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello there"

    def test_consecutive_function_calls_grouped(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Do two things"}]},
                {
                    "type": "function_call",
                    "name": "func_a",
                    "arguments": "{}",
                    "call_id": "call_1",
                },
                {
                    "type": "function_call",
                    "name": "func_b",
                    "arguments": "{}",
                    "call_id": "call_2",
                },
            ],
        }
        result = responses_to_anthropic_request(body)
        messages = result["messages"]
        # Consecutive function_calls should be in a single assistant message
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["content"]) == 2
        assert messages[1]["content"][0]["id"] == "call_1"
        assert messages[1]["content"][1]["id"] == "call_2"

    def test_temperature_and_top_p(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": "Hi",
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = responses_to_anthropic_request(body)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9


# --- responses_to_openai_chat_request ---


class TestResponsesToOpenAIChatRequest:
    def test_simple_text(self):
        body = {
            "model": "gpt-5.2",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        assert result["model"] == "gpt-5.2"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello"

    def test_instructions_to_system(self):
        body = {
            "model": "gpt-5.2",
            "instructions": "Be concise.",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be concise."
        assert result["messages"][1]["role"] == "user"

    def test_max_output_tokens(self):
        body = {
            "model": "gpt-5.2",
            "max_output_tokens": 512,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        assert result["max_tokens"] == 512

    def test_tool_calls_grouped(self):
        body = {
            "model": "gpt-5.2",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Do it"}]},
                {
                    "type": "function_call",
                    "name": "func_a",
                    "arguments": '{"x":1}',
                    "call_id": "call_1",
                },
                {
                    "type": "function_call",
                    "name": "func_b",
                    "arguments": '{"y":2}',
                    "call_id": "call_2",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result_a",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_2",
                    "output": "result_b",
                },
            ],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        messages = result["messages"]
        # user, assistant (with 2 tool_calls), tool, tool
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert len(messages[1]["tool_calls"]) == 2
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"
        assert messages[3]["role"] == "tool"
        assert messages[3]["tool_call_id"] == "call_2"

    def test_string_input(self):
        body = {"model": "gpt-5.2", "input": "Hello there"}
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello there"

    def test_tools_translation(self):
        body = {
            "model": "gpt-5.2",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            ],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search"


# --- anthropic_to_responses_response ---


class TestAnthropicToResponsesResponse:
    def test_text_response(self):
        resp = {
            "id": "msg_abc",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = anthropic_to_responses_response(resp, "claude-sonnet-4-5")
        assert result["id"] == "msg_abc"
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["model"] == "claude-sonnet-4-5"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_use_response(self):
        resp = {
            "id": "msg_abc",
            "content": [
                {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "NYC"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = anthropic_to_responses_response(resp, "claude-sonnet-4-5")
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "function_call"
        assert result["output"][0]["name"] == "get_weather"
        assert result["output"][0]["call_id"] == "toolu_123"
        assert json.loads(result["output"][0]["arguments"]) == {"location": "NYC"}

    def test_mixed_text_and_tool(self):
        resp = {
            "id": "msg_abc",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "test"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 15},
        }
        result = anthropic_to_responses_response(resp, "claude-sonnet-4-5")
        assert len(result["output"]) == 2
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["text"] == "Let me check."
        assert result["output"][1]["type"] == "function_call"

    def test_max_tokens_stop(self):
        resp = {
            "id": "msg_abc",
            "content": [{"type": "text", "text": "Partial..."}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 10, "output_tokens": 100},
        }
        result = anthropic_to_responses_response(resp, "claude-sonnet-4-5")
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"

    def test_cache_tokens(self):
        resp = {
            "id": "msg_abc",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 7,
            },
        }
        result = anthropic_to_responses_response(resp, "claude-sonnet-4-5")
        assert result["usage"]["cache_creation_input_tokens"] == 3
        assert result["usage"]["cache_read_input_tokens"] == 7


# --- openai_chat_to_responses_response ---


class TestOpenAIChatToResponsesResponse:
    def test_text_response(self):
        resp = {
            "id": "chatcmpl-abc",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = openai_chat_to_responses_response(resp, "gpt-5.2")
        assert result["object"] == "response"
        assert result["status"] == "completed"
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls_response(self):
        resp = {
            "id": "chatcmpl-abc",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":"test"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = openai_chat_to_responses_response(resp, "gpt-5.2")
        assert result["output"][0]["type"] == "function_call"
        assert result["output"][0]["call_id"] == "call_1"
        assert result["output"][0]["name"] == "search"
        assert result["status"] == "completed"

    def test_length_finish_reason(self):
        resp = {
            "id": "chatcmpl-abc",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Partial"}, "finish_reason": "length"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110},
        }
        result = openai_chat_to_responses_response(resp, "gpt-5.2")
        assert result["status"] == "incomplete"
        assert result["incomplete_details"]["reason"] == "max_output_tokens"


# --- AnthropicToResponsesStreamTranslator ---


class TestAnthropicToResponsesStreamTranslator:
    def test_text_stream_sequence(self):
        translator = AnthropicToResponsesStreamTranslator("claude-sonnet-4-5")

        # message_start
        events = translator.translate_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_abc",
                    "usage": {"input_tokens": 10},
                },
            },
        )
        assert "response.created" in events
        assert '"input_tokens": 10' in events

        # content_block_start (text)
        events = translator.translate_event(
            "content_block_start",
            {"content_block": {"type": "text", "text": ""}},
        )
        assert "response.output_item.added" in events
        assert '"type": "message"' in events

        # content_block_delta (text_delta)
        events = translator.translate_event(
            "content_block_delta",
            {"delta": {"type": "text_delta", "text": "Hello"}},
        )
        assert "response.output_text.delta" in events
        assert "Hello" in events

        # content_block_stop
        events = translator.translate_event("content_block_stop", {"index": 0})
        assert "response.output_item.done" in events

        # message_delta
        events = translator.translate_event(
            "message_delta",
            {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        )
        assert events == ""  # Only accumulates, does not emit

        # message_stop
        events = translator.translate_event("message_stop", {"type": "message_stop"})
        assert "response.completed" in events
        assert '"status": "completed"' in events

    def test_tool_call_stream_sequence(self):
        translator = AnthropicToResponsesStreamTranslator("claude-sonnet-4-5")

        translator.translate_event("message_start", {"type": "message_start", "message": {}})

        # content_block_start (tool_use)
        events = translator.translate_event(
            "content_block_start",
            {"content_block": {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {}}},
        )
        assert "response.output_item.added" in events
        assert "function_call" in events
        assert "get_weather" in events

        # content_block_delta (input_json_delta)
        events = translator.translate_event(
            "content_block_delta",
            {"delta": {"type": "input_json_delta", "partial_json": '{"loc'}},
        )
        assert "response.function_call_arguments.delta" in events

        # content_block_stop
        events = translator.translate_event("content_block_stop", {"index": 0})
        assert "response.output_item.done" in events
        assert "function_call" in events

        translator.translate_event(
            "message_delta",
            {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}},
        )
        events = translator.translate_event("message_stop", {})
        assert "response.completed" in events

    def test_flush_emits_completed_if_not_finished(self):
        translator = AnthropicToResponsesStreamTranslator("claude-sonnet-4-5")
        translator.translate_event("message_start", {"type": "message_start", "message": {}})
        translator.translate_event("content_block_start", {"content_block": {"type": "text", "text": ""}})
        translator.translate_event("content_block_delta", {"delta": {"type": "text_delta", "text": "Hi"}})
        # Stream ends without message_stop
        events = translator.flush()
        assert "response.completed" in events

    def test_flush_noop_after_completed(self):
        translator = AnthropicToResponsesStreamTranslator("claude-sonnet-4-5")
        translator.translate_event("message_start", {"type": "message_start", "message": {}})
        translator.translate_event("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {}})
        translator.translate_event("message_stop", {})
        assert translator.flush() == ""

    def test_cache_tokens(self):
        translator = AnthropicToResponsesStreamTranslator("claude-sonnet-4-5")
        events = translator.translate_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_creation_input_tokens": 5,
                        "cache_read_input_tokens": 3,
                    }
                },
            },
        )
        assert '"input_tokens": 10' in events

        translator.translate_event(
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 3,
                },
            },
        )
        events = translator.translate_event("message_stop", {})
        assert "cache_creation_input_tokens" in events
        assert "cache_read_input_tokens" in events


# --- OpenAIChatToResponsesStreamTranslator ---


class TestOpenAIChatToResponsesStreamTranslator:
    def test_text_stream_sequence(self):
        translator = OpenAIChatToResponsesStreamTranslator("gpt-5.2")

        # First chunk with role
        events = translator.translate_chunk(
            {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                "usage": {"prompt_tokens": 10},
            }
        )
        assert "response.created" in events
        assert "response.output_item.added" in events

        # Text content
        events = translator.translate_chunk(
            {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]}
        )
        assert "response.output_text.delta" in events
        assert "Hello" in events

        # Finish + usage
        events = translator.translate_chunk(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            }
        )
        assert "response.output_item.done" in events
        assert "response.completed" in events
        assert '"status": "completed"' in events

    def test_tool_call_stream_sequence(self):
        translator = OpenAIChatToResponsesStreamTranslator("gpt-5.2")

        # Init
        translator.translate_chunk(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        )

        # Tool call start
        events = translator.translate_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {"name": "search", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        assert "response.output_item.added" in events
        assert "function_call" in events

        # Tool call arguments
        events = translator.translate_chunk(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"test"}'}}]},
                        "finish_reason": None,
                    }
                ]
            }
        )
        assert "response.function_call_arguments.delta" in events

        # Finish with usage
        events = translator.translate_chunk(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        assert "response.output_item.done" in events
        assert "response.completed" in events

    def test_deferred_finish_pattern(self):
        """Usage-only chunk arrives after finish_reason chunk."""
        translator = OpenAIChatToResponsesStreamTranslator("gpt-5.2")

        translator.translate_chunk(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        )
        translator.translate_chunk({"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]})

        # Finish reason without usage
        events = translator.translate_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        # Should not emit completed yet (no usage)
        assert "response.completed" not in events

        # Usage-only chunk
        events = translator.translate_chunk(
            {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 3, "total_tokens": 45}}
        )
        assert "response.completed" in events
        assert '"input_tokens": 42' in events
        assert '"output_tokens": 3' in events

    def test_flush_emits_completed(self):
        translator = OpenAIChatToResponsesStreamTranslator("gpt-5.2")
        translator.translate_chunk(
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
        )
        translator.translate_chunk({"choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}]})
        translator.translate_chunk({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        # No usage chunk arrived, flush should emit completed
        events = translator.flush()
        assert "response.output_item.done" in events
        assert "response.completed" in events

    def test_flush_noop_after_completed(self):
        translator = OpenAIChatToResponsesStreamTranslator("gpt-5.2")
        translator.translate_chunk(
            {
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}],
                "usage": {"prompt_tokens": 5},
            }
        )
        translator.translate_chunk(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1},
            }
        )
        assert translator.flush() == ""


# --- Responses API Round-Trip Tests ---


class TestResponsesAPIRoundTrip:
    def test_responses_to_anthropic_and_back(self):
        """Responses -> Anthropic -> Responses preserves key data."""
        body = {
            "model": "claude-sonnet-4-5",
            "instructions": "Be helpful.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            "max_output_tokens": 1024,
        }

        # Responses -> Anthropic
        anthropic_req = responses_to_anthropic_request(body)
        assert anthropic_req["system"] == "Be helpful."
        assert anthropic_req["max_tokens"] == 1024
        assert anthropic_req["messages"][0]["content"] == "Hello"

        # Simulate Anthropic response
        anthropic_resp = {
            "id": "msg_abc",
            "content": [{"type": "text", "text": "Hi there!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        # Anthropic -> Responses
        responses_resp = anthropic_to_responses_response(anthropic_resp, "claude-sonnet-4-5")
        assert responses_resp["status"] == "completed"
        assert responses_resp["output"][0]["content"][0]["text"] == "Hi there!"

    def test_responses_to_openai_and_back(self):
        """Responses -> OpenAI -> Responses preserves key data."""
        body = {
            "model": "gpt-5.2",
            "instructions": "Be concise.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            ],
            "max_output_tokens": 512,
        }

        # Responses -> OpenAI
        openai_req = responses_to_openai_chat_request(body, "gpt-5.2")
        assert openai_req["messages"][0]["content"] == "Be concise."
        assert openai_req["max_tokens"] == 512
        assert openai_req["messages"][1]["content"] == "Hello"

        # Simulate OpenAI response
        openai_resp = {
            "id": "chatcmpl-abc",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        }

        # OpenAI -> Responses
        responses_resp = openai_chat_to_responses_response(openai_resp, "gpt-5.2")
        assert responses_resp["status"] == "completed"
        assert responses_resp["output"][0]["content"][0]["text"] == "Hi!"

    def test_tool_call_id_preserved_through_anthropic(self):
        """Tool call IDs preserved through Responses -> Anthropic -> Responses."""
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Weather?"}]},
                {
                    "type": "function_call",
                    "name": "get_weather",
                    "arguments": '{"city":"NYC"}',
                    "call_id": "call_orig_123",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_orig_123",
                    "output": "72°F",
                },
            ],
        }

        anthropic_req = responses_to_anthropic_request(body)
        # Verify call_id is preserved as tool_use id
        assistant_msg = anthropic_req["messages"][1]
        assert assistant_msg["content"][0]["id"] == "call_orig_123"
        # Verify call_id is preserved as tool_use_id
        user_msg = anthropic_req["messages"][2]
        assert user_msg["content"][0]["tool_use_id"] == "call_orig_123"


# --- Image Support Tests ---


class TestAnthropicToResponsesRequestImages:
    def test_base64_image(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "iVBOR..."},
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        assert len(result["input"]) == 1
        item = result["input"][0]
        assert item["role"] == "user"
        assert item["content"][0]["type"] == "input_image"
        assert item["content"][0]["image_url"] == "data:image/png;base64,iVBOR..."

    def test_url_image(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://example.com/img.png"},
                        }
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        item = result["input"][0]
        assert item["content"][0]["type"] == "input_image"
        assert item["content"][0]["image_url"] == "https://example.com/img.png"

    def test_mixed_text_and_image(self):
        body = {
            "model": "x",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc123"},
                        },
                    ],
                }
            ],
        }
        result = anthropic_to_responses_request(body, "gpt-5.2-codex")
        # Text and image become separate input items
        assert len(result["input"]) == 2
        assert result["input"][0]["content"][0]["type"] == "input_text"
        assert result["input"][1]["content"][0]["type"] == "input_image"


class TestOpenAIChatToResponsesRequestImages:
    def test_image_url_part(self):
        body = {
            "model": "gpt-5.2-codex",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc123"},
                        },
                    ],
                }
            ],
        }
        result = openai_chat_to_responses_request(body, "gpt-5.2-codex")
        assert len(result["input"]) == 1
        item = result["input"][0]
        assert item["role"] == "user"
        assert len(item["content"]) == 2
        assert item["content"][0] == {"type": "input_text", "text": "Describe this"}
        assert item["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,abc123"}


class TestResponsesToAnthropicRequestImages:
    def test_input_image_data_url(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "data:image/png;base64,iVBOR..."},
                    ],
                }
            ],
        }
        result = responses_to_anthropic_request(body)
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][0]["source"]["type"] == "base64"
        assert msg["content"][0]["source"]["media_type"] == "image/png"
        assert msg["content"][0]["source"]["data"] == "iVBOR..."

    def test_input_image_http_url(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "https://example.com/img.png"},
                    ],
                }
            ],
        }
        result = responses_to_anthropic_request(body)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][0]["source"]["type"] == "url"
        assert msg["content"][0]["source"]["url"] == "https://example.com/img.png"

    def test_mixed_text_and_image(self):
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is this?"},
                        {"type": "input_image", "image_url": "data:image/jpeg;base64,abc"},
                    ],
                }
            ],
        }
        result = responses_to_anthropic_request(body)
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2
        assert msg["content"][0] == {"type": "text", "text": "What is this?"}
        assert msg["content"][1]["type"] == "image"

    def test_text_only_returns_string(self):
        """Text-only user messages still return string content."""
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        result = responses_to_anthropic_request(body)
        assert result["messages"][0]["content"] == "Hello"


class TestResponsesToOpenAIChatRequestImages:
    def test_input_image(self):
        body = {
            "model": "gpt-5.2",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                    ],
                }
            ],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        msg = result["messages"][0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0] == {"type": "text", "text": "Describe"}
        assert msg["content"][1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}

    def test_text_only_returns_string(self):
        """Text-only returns joined string, not list."""
        body = {
            "model": "gpt-5.2",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
        }
        result = responses_to_openai_chat_request(body, "gpt-5.2")
        assert result["messages"][0]["content"] == "Hello"


class TestImageRoundTrip:
    def test_anthropic_to_openai_to_anthropic_preserves_image(self):
        """Image data preserved through Anthropic -> Responses -> Anthropic."""
        from claudegate.copilot_translate import _translate_content_to_openai
        from claudegate.openai_translate import openai_to_anthropic_request

        # Start with Anthropic format
        openai_body = {
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

        # OpenAI -> Anthropic
        anthropic_req = openai_to_anthropic_request(openai_body)
        msg = anthropic_req["messages"][0]
        assert msg["content"][0] == {"type": "text", "text": "What is this?"}
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"]["data"] == "iVBOR..."

        # Anthropic -> OpenAI (via _translate_content_to_openai)
        result = _translate_content_to_openai(msg["content"])
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "What is this?"}
        assert result[1]["image_url"]["url"] == "data:image/png;base64,iVBOR..."

    def test_responses_image_round_trip(self):
        """Image preserved through Responses -> Anthropic -> Responses."""
        body = {
            "model": "claude-sonnet-4-5",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                    ],
                }
            ],
        }
        # Responses -> Anthropic
        anthropic_req = responses_to_anthropic_request(body)
        msg = anthropic_req["messages"][0]
        assert msg["content"][0] == {"type": "text", "text": "Describe"}
        assert msg["content"][1]["type"] == "image"
        assert msg["content"][1]["source"]["data"] == "abc123"

        # Anthropic -> Responses
        anthropic_body = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": msg["content"]}],
        }
        responses_req = anthropic_to_responses_request(anthropic_body, "gpt-5.2")
        # The text and image should be in separate input items
        text_item = responses_req["input"][0]
        image_item = responses_req["input"][1]
        assert text_item["content"][0]["type"] == "input_text"
        assert image_item["content"][0]["type"] == "input_image"
        assert "abc123" in image_item["content"][0]["image_url"]
