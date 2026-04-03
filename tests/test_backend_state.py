"""Tests for claudegate/backend_state.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claudegate.backend_state import BackendState, parse_backend_string

# --- parse_backend_string ---


class TestParseBackendString:
    def test_single_copilot(self):
        assert parse_backend_string("copilot") == ("copilot", "")

    def test_single_bedrock(self):
        assert parse_backend_string("bedrock") == ("bedrock", "")

    def test_copilot_bedrock_fallback(self):
        assert parse_backend_string("copilot,bedrock") == ("copilot", "bedrock")

    def test_bedrock_copilot_fallback(self):
        assert parse_backend_string("bedrock,copilot") == ("bedrock", "copilot")

    def test_whitespace_handling(self):
        assert parse_backend_string("  copilot , bedrock  ") == ("copilot", "bedrock")

    def test_case_insensitive(self):
        assert parse_backend_string("COPILOT") == ("copilot", "")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_backend_string("")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid backend"):
            parse_backend_string("invalid")

    def test_invalid_fallback_raises(self):
        with pytest.raises(ValueError, match="Invalid fallback backend"):
            parse_backend_string("copilot,invalid")

    def test_same_primary_and_fallback_raises(self):
        with pytest.raises(ValueError, match="Fallback cannot be the same"):
            parse_backend_string("copilot,copilot")

    def test_three_backends_raises(self):
        with pytest.raises(ValueError, match="At most two backends"):
            parse_backend_string("copilot,bedrock,copilot")


# --- BackendState ---


class TestBackendState:
    def test_initial_state(self):
        bs = BackendState("copilot", "bedrock")
        assert bs.primary == "copilot"
        assert bs.fallback == "bedrock"
        assert bs.copilot_backend is None
        assert bs.copilot_usage_cache is None

    def test_initial_state_no_fallback(self):
        bs = BackendState("bedrock")
        assert bs.primary == "bedrock"
        assert bs.fallback == ""

    def test_set_copilot_backend(self):
        bs = BackendState("copilot")
        mock_backend = MagicMock()
        mock_cache = MagicMock()
        bs.set_copilot_backend(mock_backend, mock_cache)
        assert bs.copilot_backend is mock_backend
        assert bs.copilot_usage_cache is mock_cache


class TestBackendStateSwitch:
    @pytest.mark.asyncio
    async def test_switch_no_change(self):
        bs = BackendState("copilot", "bedrock")
        result = await bs.switch("copilot", "bedrock")
        assert result["changed"] is False
        assert result["primary"] == "copilot"
        assert result["fallback"] == "bedrock"

    @pytest.mark.asyncio
    async def test_switch_to_bedrock(self):
        bs = BackendState("copilot")
        bs._copilot_backend = MagicMock()
        result = await bs.switch("bedrock")
        assert result["changed"] is True
        assert bs.primary == "bedrock"
        assert bs.fallback == ""

    @pytest.mark.asyncio
    async def test_switch_to_copilot_with_existing_backend(self):
        bs = BackendState("bedrock")
        bs._copilot_backend = MagicMock()
        result = await bs.switch("copilot")
        assert result["changed"] is True
        assert bs.primary == "copilot"

    @pytest.mark.asyncio
    async def test_switch_to_copilot_lazy_init(self):
        """Switching TO copilot when not initialized triggers _init_copilot."""
        bs = BackendState("bedrock")
        assert bs.copilot_backend is None

        with patch.object(bs, "_init_copilot", new_callable=AsyncMock) as mock_init:
            result = await bs.switch("copilot")
            mock_init.assert_awaited_once()
        assert result["changed"] is True
        assert bs.primary == "copilot"

    @pytest.mark.asyncio
    async def test_switch_copilot_fallback_lazy_init(self):
        """Switching with copilot as fallback also triggers lazy init."""
        bs = BackendState("copilot")
        bs._copilot_backend = MagicMock()
        # Now switch to bedrock primary, copilot fallback — copilot already initialized
        result = await bs.switch("bedrock", "copilot")
        assert result["changed"] is True

    @pytest.mark.asyncio
    async def test_switch_invalid_primary(self):
        bs = BackendState("copilot")
        with pytest.raises(ValueError, match="Invalid backend"):
            await bs.switch("invalid")

    @pytest.mark.asyncio
    async def test_switch_invalid_fallback(self):
        bs = BackendState("copilot")
        with pytest.raises(ValueError, match="Invalid fallback"):
            await bs.switch("copilot", "invalid")

    @pytest.mark.asyncio
    async def test_switch_same_primary_and_fallback(self):
        bs = BackendState("copilot")
        with pytest.raises(ValueError, match="Fallback cannot be the same"):
            await bs.switch("bedrock", "bedrock")

    @pytest.mark.asyncio
    async def test_switch_init_failure_raises(self):
        """If _init_copilot fails, ValueError is raised and state is unchanged."""
        bs = BackendState("bedrock")

        with (
            patch.object(bs, "_init_copilot", new_callable=AsyncMock, side_effect=ValueError("auth failed")),
            pytest.raises(ValueError, match="auth failed"),
        ):
            await bs.switch("copilot")
        # State should remain unchanged
        assert bs.primary == "bedrock"


class TestBackendStateClose:
    @pytest.mark.asyncio
    async def test_close_with_copilot(self):
        bs = BackendState("copilot")
        mock_backend = AsyncMock()
        mock_cache = AsyncMock()
        bs._copilot_backend = mock_backend
        bs._copilot_usage_cache = mock_cache
        await bs.close()
        mock_backend.close.assert_awaited_once()
        mock_cache.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_copilot(self):
        bs = BackendState("bedrock")
        await bs.close()  # Should not raise


class TestBackendStateInitCopilot:
    @pytest.mark.asyncio
    async def test_init_copilot_success(self):
        bs = BackendState("bedrock")

        mock_backend = AsyncMock()
        mock_backend.list_models = AsyncMock(return_value=[{"id": "m1"}])
        mock_cache = MagicMock()

        with (
            patch("claudegate.copilot_auth.get_github_token", return_value="tok"),
            patch("claudegate.copilot_client.CopilotBackend", return_value=mock_backend),
            patch("claudegate.copilot_usage.CopilotUsageCache", return_value=mock_cache),
            patch("claudegate.models.set_copilot_models") as mock_set,
        ):
            await bs._init_copilot()

        assert bs.copilot_backend is mock_backend
        assert bs.copilot_usage_cache is mock_cache
        mock_set.assert_called_once_with([{"id": "m1"}])

    @pytest.mark.asyncio
    async def test_init_copilot_no_token(self):
        bs = BackendState("bedrock")

        with (
            patch("claudegate.copilot_auth.get_github_token", side_effect=RuntimeError("no token")),
            pytest.raises(ValueError, match="Failed to get GitHub token"),
        ):
            await bs._init_copilot()
