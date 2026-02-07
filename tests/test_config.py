"""Tests for claudegate/config.py."""

from claudegate.config import (
    DEFAULT_AWS_REGION,
    DEFAULT_BEDROCK_REGION_PREFIX,
    DEFAULT_HOST,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PORT,
    DEFAULT_READ_TIMEOUT,
    LOGGING_CONFIG,
    logger,
)


class TestDefaultValues:
    def test_default_host(self):
        assert DEFAULT_HOST == "0.0.0.0"

    def test_default_port(self):
        assert DEFAULT_PORT == 8080

    def test_default_log_level(self):
        assert DEFAULT_LOG_LEVEL == "INFO"

    def test_default_aws_region(self):
        assert DEFAULT_AWS_REGION == "us-west-2"

    def test_default_bedrock_region_prefix(self):
        assert DEFAULT_BEDROCK_REGION_PREFIX == "us"

    def test_default_read_timeout(self):
        assert DEFAULT_READ_TIMEOUT == 300


class TestLoggingConfig:
    def test_version(self):
        assert LOGGING_CONFIG["version"] == 1

    def test_has_expected_keys(self):
        assert "formatters" in LOGGING_CONFIG
        assert "handlers" in LOGGING_CONFIG
        assert "loggers" in LOGGING_CONFIG

    def test_claudegate_logger_configured(self):
        assert "claudegate" in LOGGING_CONFIG["loggers"]

    def test_formatters(self):
        assert "default" in LOGGING_CONFIG["formatters"]
        assert "access" in LOGGING_CONFIG["formatters"]

    def test_handlers(self):
        assert "default" in LOGGING_CONFIG["handlers"]
        assert "access" in LOGGING_CONFIG["handlers"]


class TestLogger:
    def test_logger_name(self):
        assert logger.name == "claudegate"
