"""Tests for claudegate/models.py."""

import pytest

from claudegate.models import (
    BEDROCK_MODEL_MAP,
    COPILOT_MODEL_MAP,
    COPILOT_OPENAI_MODEL_MAP,
    DEFAULT_MODEL,
    add_region_prefix,
    get_available_copilot_models,
    get_bedrock_model,
    get_copilot_model,
    get_copilot_openai_model,
    is_claude_model,
    model_requires_responses_api,
    set_copilot_models,
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
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the dynamic model registry before and after each test."""
        set_copilot_models([])
        yield
        set_copilot_models([])

    @pytest.mark.parametrize("anthropic_name,copilot_id", list(COPILOT_MODEL_MAP.items()))
    def test_all_map_entries(self, anthropic_name, copilot_id):
        model_id, returned_name = get_copilot_model(anthropic_name)
        assert model_id == copilot_id
        assert returned_name == anthropic_name

    def test_partial_match(self):
        model_id, returned_name = get_copilot_model("prefix-claude-sonnet-4-5-20250929-suffix")
        assert model_id == "claude-sonnet-4.5"
        assert returned_name == "claude-sonnet-4-5-20250929"

    def test_openai_model_direct_match(self):
        model_id, returned_name = get_copilot_model("gpt-5.1-codex")
        assert model_id == "gpt-5.1-codex"
        assert returned_name == "gpt-5.1-codex"

    def test_openai_model_gemini(self):
        model_id, returned_name = get_copilot_model("gemini-2.5-pro")
        assert model_id == "gemini-2.5-pro"
        assert returned_name == "gemini-2.5-pro"

    def test_openai_model_gpt4o(self):
        model_id, returned_name = get_copilot_model("gpt-4o")
        assert model_id == "gpt-4o"
        assert returned_name == "gpt-4o"

    def test_openai_model_partial_match(self):
        model_id, returned_name = get_copilot_model("prefix-gpt-4o-suffix")
        assert model_id == "gpt-4o"
        assert returned_name == "gpt-4o"

    def test_unknown_non_claude_passes_through(self):
        model_id, returned_name = get_copilot_model("totally-unknown")
        assert model_id == "totally-unknown"
        assert returned_name == "totally-unknown"

    def test_smart_fallback_sonnet_4_6(self):
        """Regression: claude-sonnet-4-6 was incorrectly matching claude-sonnet-4."""
        set_copilot_models(
            [
                {"id": "claude-sonnet-4.5"},
                {"id": "claude-sonnet-4"},
                {"id": "claude-haiku-4.5"},
            ]
        )

        # Without date suffix
        model_id, returned_name = get_copilot_model("claude-sonnet-4-6")
        assert model_id == "claude-sonnet-4.5"
        assert returned_name == "claude-sonnet-4-6"

        # With date suffix
        model_id, returned_name = get_copilot_model("claude-sonnet-4-6-20250115")
        assert model_id == "claude-sonnet-4.5"
        assert returned_name == "claude-sonnet-4-6-20250115"

    def test_smart_fallback_all_families(self):
        """Smart fallback works independently for opus, sonnet, and haiku."""
        set_copilot_models(
            [
                {"id": "claude-opus-4.5"},
                {"id": "claude-sonnet-4.6"},
                {"id": "claude-haiku-4.5"},
            ]
        )

        # Opus: 4.6 unavailable, falls back to 4.5
        model_id, _ = get_copilot_model("claude-opus-4-6")
        assert model_id == "claude-opus-4.5"

        # Sonnet: 4.6 available, exact match
        model_id, _ = get_copilot_model("claude-sonnet-4-6")
        assert model_id == "claude-sonnet-4.6"

        # Haiku: future 5.0 unavailable, falls back to 4.5
        model_id, _ = get_copilot_model("claude-haiku-5-0")
        assert model_id == "claude-haiku-4.5"

    def test_smart_fallback_unknown_family(self):
        """Smart fallback works for model families not in any hardcoded map."""
        set_copilot_models(
            [
                {"id": "claude-couplet-2.1"},
                {"id": "claude-couplet-1.5"},
                {"id": "claude-quatrain-3.2"},
            ]
        )

        # Fallback to newest available
        model_id, returned_name = get_copilot_model("claude-couplet-3-0")
        assert model_id == "claude-couplet-2.1"
        assert returned_name == "claude-couplet-3-0"

        # Exact match
        model_id, _ = get_copilot_model("claude-quatrain-3-2")
        assert model_id == "claude-quatrain-3.2"

    def test_smart_fallback_version_lower_than_available(self):
        """When requested version is older than all available, use newest."""
        set_copilot_models(
            [
                {"id": "claude-sonnet-10.7"},
                {"id": "claude-sonnet-9.12"},
            ]
        )

        model_id, _ = get_copilot_model("claude-sonnet-8-0")
        assert model_id == "claude-sonnet-10.7"

    def test_hardcoded_map_trusted_when_not_in_dynamic_registry(self):
        """Hardcoded map target is trusted even if not yet in the dynamic registry."""
        # Simulate startup timing: dynamic models loaded but incomplete
        set_copilot_models([{"id": "gpt-4o"}])

        # claude-sonnet-4-5 is in COPILOT_MODEL_MAP -> claude-sonnet-4.5,
        # but claude-sonnet-4.5 is NOT in the dynamic registry.
        # Should still return the hardcoded target, not fall to default.
        model_id, returned_name = get_copilot_model("claude-sonnet-4-5")
        assert model_id == "claude-sonnet-4.5"
        assert returned_name == "claude-sonnet-4-5"

    def test_empty_string_passes_through(self):
        model_id, returned_name = get_copilot_model("")
        assert model_id == ""
        assert returned_name == ""


# --- get_copilot_openai_model ---


class TestGetCopilotOpenAIModel:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the dynamic model registry before and after each test."""
        set_copilot_models([])
        yield
        set_copilot_models([])

    @pytest.mark.parametrize("model_name,expected", list(COPILOT_OPENAI_MODEL_MAP.items()))
    def test_all_openai_map_entries(self, model_name, expected):
        assert get_copilot_openai_model(model_name) == expected

    def test_direct_openai_model(self):
        assert get_copilot_openai_model("gpt-4o") == "gpt-4o"

    def test_copilot_display_name(self):
        assert get_copilot_openai_model("claude-sonnet-4.5") == "claude-sonnet-4.5"

    def test_anthropic_name_mapping(self):
        # Anthropic versioned name should map via COPILOT_MODEL_MAP (no dynamic models)
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
        assert get_copilot_openai_model("gemini-3-pro-preview") == "gemini-3-pro-preview"


# --- Dynamic Copilot Model Registry ---


class TestDynamicCopilotModels:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the dynamic model registry before and after each test."""
        set_copilot_models([])
        yield
        set_copilot_models([])

    def test_set_and_get_copilot_models(self):
        models = [{"id": "model-a"}, {"id": "model-b"}]
        set_copilot_models(models)
        assert get_available_copilot_models() == models

    def test_get_empty_when_not_set(self):
        assert get_available_copilot_models() == []

    def test_dynamic_model_exact_match(self):
        set_copilot_models([{"id": "gemini-2.5-pro-preview"}, {"id": "gpt-4o-2025-01"}])
        assert get_copilot_openai_model("gemini-2.5-pro-preview") == "gemini-2.5-pro-preview"
        assert get_copilot_openai_model("gpt-4o-2025-01") == "gpt-4o-2025-01"

    def test_dynamic_model_takes_priority_over_hardcoded(self):
        """Dynamic registry is checked before hardcoded maps."""
        set_copilot_models([{"id": "gpt-4o"}])
        assert get_copilot_openai_model("gpt-4o") == "gpt-4o"

    def test_hardcoded_not_used_when_dynamic_populated(self):
        """When dynamic models exist, hardcoded map is skipped to avoid wrong IDs."""
        set_copilot_models([{"id": "some-new-model"}])
        # gpt-4o is in COPILOT_OPENAI_MODEL_MAP but NOT in dynamic registry;
        # with dynamic models loaded, it passes through as-is (not via hardcoded map)
        assert get_copilot_openai_model("gpt-4o") == "gpt-4o"

    def test_unknown_model_not_validated_by_hardcoded(self):
        """When dynamic models exist, hardcoded map is not consulted."""
        set_copilot_models([{"id": "gemini-2.5-pro"}, {"id": "claude-sonnet-4.5"}])
        # gemini-3-pro-preview is in COPILOT_OPENAI_MODEL_MAP but NOT in dynamic registry;
        # it should pass through as-is, not be "validated" by the hardcoded map
        assert get_copilot_openai_model("gemini-3-pro-preview") == "gemini-3-pro-preview"

    def test_unknown_model_passthrough_with_dynamic(self):
        """Unknown models pass through when dynamic registry is populated."""
        set_copilot_models([{"id": "model-a"}])
        assert get_copilot_openai_model("totally-unknown") == "totally-unknown"

    def test_anthropic_name_resolves_when_target_in_dynamic(self):
        """Anthropic versioned names resolve via COPILOT_MODEL_MAP when target is in dynamic registry."""
        set_copilot_models([{"id": "claude-sonnet-4.5"}])
        assert get_copilot_openai_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4.5"

    def test_anthropic_name_passthrough_when_target_not_in_dynamic(self):
        """Anthropic names pass through as-is when resolved target is not in dynamic registry."""
        set_copilot_models([{"id": "some-model"}])
        # claude-sonnet-4.5 not in dynamic registry, so versioned name passes through
        assert get_copilot_openai_model("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5-20250929"

    def test_anthropic_partial_match_resolves_when_in_dynamic(self):
        """Partial Anthropic name matches resolve when target is in dynamic registry."""
        set_copilot_models([{"id": "claude-sonnet-4.5"}])
        assert get_copilot_openai_model("prefix-claude-sonnet-4-5-20250929-suffix") == "claude-sonnet-4.5"

    def test_openai_smart_fallback_for_claude_models(self):
        """Test that OpenAI endpoint also gets smart fallback for Claude models."""
        # Mock available models without Sonnet 4.6
        set_copilot_models(
            [
                {"id": "claude-sonnet-4.5"},
                {"id": "claude-opus-4.5"},
                {"id": "gpt-4o"},
            ]
        )

        # Test Sonnet 4.6 fallback
        assert get_copilot_openai_model("claude-sonnet-4-6") == "claude-sonnet-4.5"

        # Test Opus 4.6 fallback
        assert get_copilot_openai_model("claude-opus-4-6") == "claude-opus-4.5"

        # Non-Claude models should pass through
        assert get_copilot_openai_model("gpt-4o") == "gpt-4o"

    def test_openai_substring_bug_regression(self):
        """Regression: partial match must not match claude-sonnet-4 for claude-sonnet-4-6."""
        set_copilot_models(
            [
                {"id": "claude-sonnet-4.5"},
                {"id": "claude-sonnet-4"},
            ]
        )

        # Should get 4.5 via smart fallback, NOT 4 via substring match
        assert get_copilot_openai_model("claude-sonnet-4-6") == "claude-sonnet-4.5"

    def test_get_copilot_model_checks_dynamic_registry(self):
        """get_copilot_model() finds models in the dynamic registry."""
        from claudegate.models import get_copilot_model

        set_copilot_models([{"id": "gemini-2.5-pro"}, {"id": "gpt-4o"}])
        model_id, name = get_copilot_model("gemini-2.5-pro")
        assert model_id == "gemini-2.5-pro"
        assert name == "gemini-2.5-pro"

    def test_get_copilot_model_non_claude_passthrough_dynamic(self):
        """Non-Claude models not in any map pass through as-is with dynamic models."""
        set_copilot_models([{"id": "claude-sonnet-4.5"}, {"id": "some-custom-model"}])
        model_id, name = get_copilot_model("some-custom-model")
        assert model_id == "some-custom-model"
        assert name == "some-custom-model"

    def test_get_copilot_model_non_claude_passthrough_static(self):
        """Non-Claude models not in any map pass through as-is without dynamic models."""
        model_id, name = get_copilot_model("my-custom-llm")
        assert model_id == "my-custom-llm"
        assert name == "my-custom-llm"

    def test_get_copilot_model_unknown_claude_defaults(self):
        """Unknown Claude models still default to the Claude default."""
        model_id, name = get_copilot_model("claude-unknown-99-0")
        # Falls through Claude-specific lookups, but is_claude_model is True,
        # so it hits the Claude default
        assert model_id == "claude-sonnet-4.5"
        assert name == "claude-sonnet-4-5-20250929"


# --- is_claude_model ---


class TestIsClaudeModel:
    def test_claude_anthropic_names(self):
        assert is_claude_model("claude-sonnet-4-5-20250929") is True
        assert is_claude_model("claude-opus-4-6") is True
        assert is_claude_model("claude-haiku-4-5") is True
        assert is_claude_model("claude-3-5-sonnet-20241022") is True

    def test_bedrock_model_ids(self):
        assert is_claude_model("anthropic.claude-sonnet-4-5-20250929-v1:0") is True
        assert is_claude_model("us.anthropic.claude-3-haiku-20240307-v1:0") is True

    def test_non_claude_models(self):
        assert is_claude_model("gpt-4o") is False
        assert is_claude_model("gpt-5.1-codex") is False
        assert is_claude_model("gemini-2.5-pro") is False
        assert is_claude_model("grok-code-fast-1") is False
        assert is_claude_model("totally-unknown") is False

    def test_empty_string(self):
        assert is_claude_model("") is False


# --- Prefix Stripping Integration ---


class TestPrefixStrippingIntegration:
    """Test that provider prefixes (e.g. github-copilot/) are stripped in all lookup functions."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        set_copilot_models([])
        yield
        set_copilot_models([])

    def test_get_copilot_model_strips_prefix_claude(self):
        model_id, name = get_copilot_model("github-copilot/claude-sonnet-4-5")
        assert model_id == "claude-sonnet-4.5"
        assert name == "claude-sonnet-4-5"

    def test_get_copilot_model_strips_prefix_openai(self):
        model_id, name = get_copilot_model("github-copilot/gpt-5.3-codex")
        assert model_id == "gpt-5.3-codex"
        assert name == "gpt-5.3-codex"

    def test_get_copilot_openai_model_strips_prefix(self):
        assert get_copilot_openai_model("github-copilot/gpt-5.3-codex") == "gpt-5.3-codex"

    def test_get_copilot_openai_model_strips_prefix_claude(self):
        assert get_copilot_openai_model("github-copilot/claude-sonnet-4-5-20250929") == "claude-sonnet-4.5"

    def test_get_bedrock_model_strips_prefix(self, monkeypatch):
        monkeypatch.setattr("claudegate.models.BEDROCK_REGION_PREFIX", "")
        result = get_bedrock_model("github-copilot/claude-sonnet-4-5-20250929")
        assert result == "anthropic.claude-sonnet-4-5-20250929-v1:0"

    def test_dynamic_registry_with_prefix(self):
        """Prefix stripping works together with dynamic registry lookups."""
        set_copilot_models([{"id": "claude-sonnet-4.5"}, {"id": "gpt-5.3-codex"}])

        model_id, name = get_copilot_model("github-copilot/claude-sonnet-4-5")
        assert model_id == "claude-sonnet-4.5"
        assert name == "claude-sonnet-4-5"

        assert get_copilot_openai_model("github-copilot/gpt-5.3-codex") == "gpt-5.3-codex"

    def test_unknown_prefix_not_stripped(self):
        """Only known prefixes (github-copilot/) are stripped; unknown prefixes pass through."""
        # Non-Claude model with unknown prefix — passes through as-is
        result = get_copilot_openai_model("custom-provider/some-model")
        assert result == "custom-provider/some-model"

    def test_slash_in_model_name_preserved(self):
        """Model names containing slashes that aren't known prefixes are preserved."""
        result = get_copilot_openai_model("org/model-name")
        assert result == "org/model-name"


# --- model_requires_responses_api ---


class TestModelRequiresResponsesApi:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        set_copilot_models([])
        yield
        set_copilot_models([])

    def test_codex_model_responses_only(self):
        """Codex models that only support /responses return True."""
        set_copilot_models(
            [
                {"id": "gpt-5.2-codex", "supported_endpoints": ["/responses"]},
            ]
        )
        assert model_requires_responses_api("gpt-5.2-codex") is True

    def test_dual_endpoint_model(self):
        """Models supporting both /chat/completions and /responses return False."""
        set_copilot_models(
            [
                {"id": "gpt-5.2", "supported_endpoints": ["/chat/completions", "/responses"]},
            ]
        )
        assert model_requires_responses_api("gpt-5.2") is False

    def test_chat_completions_only(self):
        """Models supporting only /chat/completions return False."""
        set_copilot_models(
            [
                {"id": "claude-sonnet-4.5", "supported_endpoints": ["/chat/completions", "/v1/messages"]},
            ]
        )
        assert model_requires_responses_api("claude-sonnet-4.5") is False

    def test_no_metadata(self):
        """Models with no supported_endpoints field return False."""
        set_copilot_models(
            [
                {"id": "gpt-4o"},
            ]
        )
        assert model_requires_responses_api("gpt-4o") is False

    def test_unknown_model(self):
        """Models not in the registry return False."""
        set_copilot_models(
            [
                {"id": "gpt-5.2-codex", "supported_endpoints": ["/responses"]},
            ]
        )
        assert model_requires_responses_api("totally-unknown") is False

    def test_empty_registry(self):
        """Empty registry returns False for any model."""
        assert model_requires_responses_api("gpt-5.2-codex") is False

    def test_invalid_supported_endpoints_ignored(self):
        """Non-list endpoint metadata is ignored safely."""
        set_copilot_models([{"id": "gpt-5.2-codex", "supported_endpoints": "/responses"}])
        assert model_requires_responses_api("gpt-5.2-codex") is False

    def test_non_string_endpoints_filtered(self):
        """Endpoint list keeps only string entries."""
        set_copilot_models(
            [
                {"id": "gpt-5.2-codex", "supported_endpoints": ["/responses", 123, None]},
            ]
        )
        assert model_requires_responses_api("gpt-5.2-codex") is True

    def test_chat_endpoint_with_invalid_entries(self):
        """Valid /chat/completions entry still disables responses-only routing."""
        set_copilot_models(
            [
                {"id": "gpt-5.2", "supported_endpoints": ["/responses", {}, "/chat/completions"]},
            ]
        )
        assert model_requires_responses_api("gpt-5.2") is False
