"""Tests for claudegate/client.py."""

from unittest.mock import MagicMock, patch

from claudegate.client import get_bedrock_client, reset_bedrock_client


class TestGetBedrockClient:
    def test_creates_client_on_first_call(self):
        mock_client = MagicMock()
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            result = get_bedrock_client()
            assert result is mock_client
            mock_boto3.client.assert_called_once()

    def test_returns_singleton(self):
        mock_client = MagicMock()
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            first = get_bedrock_client()
            second = get_bedrock_client()
            assert first is second
            # boto3.client only called once due to singleton
            assert mock_boto3.client.call_count == 1

    def test_env_region(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            get_bedrock_client()
            call_kwargs = mock_boto3.client.call_args
            assert call_kwargs.kwargs["region_name"] == "eu-west-1"

    def test_default_region(self):
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            get_bedrock_client()
            call_kwargs = mock_boto3.client.call_args
            assert call_kwargs.kwargs["region_name"] == "us-west-2"

    def test_env_timeout(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_READ_TIMEOUT", "600")
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            get_bedrock_client()
            config = mock_boto3.client.call_args.kwargs["config"]
            assert config.read_timeout == 600

    def test_default_timeout(self):
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            get_bedrock_client()
            config = mock_boto3.client.call_args.kwargs["config"]
            assert config.read_timeout == 300

    def test_retries_disabled(self):
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = MagicMock()
            get_bedrock_client()
            config = mock_boto3.client.call_args.kwargs["config"]
            assert config.retries == {"max_attempts": 0}


class TestResetBedrockClient:
    def test_clears_singleton(self):
        mock_client = MagicMock()
        with patch("claudegate.client.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_client
            get_bedrock_client()
            reset_bedrock_client()
            # After reset, next call should create a new client
            mock_client2 = MagicMock()
            mock_boto3.client.return_value = mock_client2
            result = get_bedrock_client()
            assert result is mock_client2
