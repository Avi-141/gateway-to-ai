"""Tests for claudegate/models.py."""

import pytest

from claudegate.models import (
    BEDROCK_MODEL_MAP,
    COPILOT_MODEL_MAP,
    COPILOT_OPENAI_MODEL_MAP,
    DEFAULT_COPILOT_MODEL,
    DEFAULT_MODEL,
    add_region_prefix,
    get_bedrock_model,
    get_copilot_model,
    get_copilot_openai_model,
)

# --- add_region_prefix ---


class TestAddRegionPrefix:
    def test_with_prefix(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "us")
        assert (
            add_region_prefix("anthropic.claude-3-haiku-20240307-v1:0") == "us.anthropic.claude-3-haiku-20240307-v1:0"
        )

    def test_empty_prefix(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        assert add_region_prefix("anthropic.claude-3-haiku-20240307-v1:0") == "anthropic.claude-3-haiku-20240307-v1:0"

    def test_no_prefix_none(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        model = "some-model"
        assert add_region_prefix(model) == model


# --- get_bedrock_model ---


class TestGetBedrockModel:
    @pytest.mark.parametrize("anthropic_name,bedrock_id", list(BEDROCK_MODEL_MAP.items()))
    def test_all_map_entries(self, anthropic_name, bedrock_id, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        assert get_bedrock_model(anthropic_name) == bedrock_id

    def test_direct_match_with_prefix(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "eu")
        result = get_bedrock_model("claude-sonnet-4-5-20250929")
        assert result == "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_passthrough_bedrock_model_id(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "us")
        # Already contains "anthropic." so should pass through as-is
        model = "us.anthropic.claude-3-haiku-20240307-v1:0"
        assert get_bedrock_model(model) == model

    def test_partial_match(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        # A model name that contains a known key as substring
        result = get_bedrock_model("some-prefix-claude-sonnet-4-5-20250929-suffix")
        assert result == "anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_unknown_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        result = get_bedrock_model("completely-unknown-model")
        assert result == DEFAULT_MODEL

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        result = get_bedrock_model("")
        assert result == DEFAULT_MODEL


# --- get_copilot_model ---


class TestGetCopilotModel:
    @pytest.mark.parametrize("anthropic_name,copilot_id", list(COPILOT_MODEL_MAP.items()))
    def test_all_map_entries(self, anthropic_name, copilot_id):
        model_id, returned_name = get_copilot_model(anthropic_name)
        assert model_id == copilot_id
        assert returned_name == anthropic_name

    def test_partial_match(self):
        model_id, returned_name = get_copilot_model("prefix-claude-sonnet-4-5-20250929-suffix")
        assert model_id == "claude-sonnet-4.5"
        assert returned_name == "claude-sonnet-4-5-20250929"

    def test_unknown_falls_back_to_default(self):
        model_id, returned_name = get_copilot_model("totally-unknown")
        assert model_id == DEFAULT_COPILOT_MODEL
        assert returned_name == "claude-sonnet-4-5-20250929"

    def test_empty_string_falls_back_to_default(self):
        model_id, returned_name = get_copilot_model("")
        assert model_id == DEFAULT_COPILOT_MODEL
        assert returned_name == "claude-sonnet-4-5-20250929"


# --- get_copilot_openai_model ---


class TestGetCopilotOpenAIModel:
    @pytest.mark.parametrize("model_name,expected", list(COPILOT_OPENAI_MODEL_MAP.items()))
    def test_all_openai_map_entries(self, model_name, expected):
        assert get_copilot_openai_model(model_name) == expected

    def test_direct_openai_model(self):
        assert get_copilot_openai_model("gpt-4o") == "gpt-4o"

    def test_copilot_display_name(self):
        assert get_copilot_openai_model("claude-sonnet-4.5") == "claude-sonnet-4.5"

    def test_anthropic_name_mapping(self):
        # Anthropic versioned name should map via COPILOT_MODEL_MAP
        assert get_copilot_openai_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4.5"

    def test_anthropic_partial_match(self):
        assert get_copilot_openai_model("prefix-claude-sonnet-4-5-20250929-suffix") == "claude-sonnet-4.5"

    def test_unknown_model_passthrough(self):
        assert get_copilot_openai_model("some-custom-model") == "some-custom-model"

    def test_empty_string_passthrough(self):
        assert get_copilot_openai_model("") == ""

    def test_gpt5_codex_models(self):
        assert get_copilot_openai_model("gpt-5.1-codex") == "gpt-5.1-codex"
        assert get_copilot_openai_model("gpt-5.1-codex-max") == "gpt-5.1-codex-max"
        assert get_copilot_openai_model("gpt-5.2-codex") == "gpt-5.2-codex"

    def test_gemini_models(self):
        assert get_copilot_openai_model("gemini-2.5-pro") == "gemini-2.5-pro"
        assert get_copilot_openai_model("gemini-3-pro") == "gemini-3-pro"
