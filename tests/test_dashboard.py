"""Tests for the dashboard routes and supporting modules."""

import logging
import subprocess
from unittest.mock import patch

import pytest


class TestRingBufferHandler:
    """Tests for the in-memory ring buffer log handler."""

    def test_captures_log_entries(self):
        from claudegate.log_buffer import RingBufferHandler

        handler = RingBufferHandler(maxlen=10)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.ring_buffer")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.info("hello")
            logger.warning("world")

            entries = handler.get_entries()
            assert len(entries) == 2
            assert entries[0]["level"] == "INFO"
            assert entries[0]["message"] == "hello"
            assert entries[1]["level"] == "WARNING"
            assert entries[1]["message"] == "world"
            assert "timestamp" in entries[0]
            assert entries[0]["logger"] == "test.ring_buffer"
        finally:
            logger.removeHandler(handler)

    def test_respects_maxlen(self):
        from claudegate.log_buffer import RingBufferHandler

        handler = RingBufferHandler(maxlen=3)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.ring_buffer_maxlen")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            for i in range(5):
                logger.info(f"msg-{i}")

            entries = handler.get_entries()
            assert len(entries) == 3
            assert entries[0]["message"] == "msg-2"
            assert entries[2]["message"] == "msg-4"
        finally:
            logger.removeHandler(handler)

    def test_level_filter(self):
        from claudegate.log_buffer import RingBufferHandler

        handler = RingBufferHandler(maxlen=10)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.ring_buffer_filter")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            logger.debug("dbg")
            logger.info("inf")
            logger.warning("warn")
            logger.error("err")

            entries = handler.get_entries(level_filter="WARNING")
            assert len(entries) == 2
            assert entries[0]["level"] == "WARNING"
            assert entries[1]["level"] == "ERROR"
        finally:
            logger.removeHandler(handler)

    def test_limit(self):
        from claudegate.log_buffer import RingBufferHandler

        handler = RingBufferHandler(maxlen=100)
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger("test.ring_buffer_limit")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        try:
            for i in range(10):
                logger.info(f"msg-{i}")

            entries = handler.get_entries(limit=3)
            assert len(entries) == 3
            assert entries[0]["message"] == "msg-7"
        finally:
            logger.removeHandler(handler)


class TestGetServiceStatus:
    """Tests for the structured service status function."""

    @patch("claudegate.service._detect_platform", return_value="macos")
    @patch("claudegate.service._plist_path")
    @patch("claudegate.service._launchd_pid", return_value=12345)
    def test_macos_installed_running(self, _mock_pid, mock_path, _mock_plat):
        from claudegate.service import get_service_status

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__str__ = lambda self: "/Users/test/Library/LaunchAgents/com.claudegate.plist"

        status = get_service_status()
        assert status["platform"] == "macos"
        assert status["installed"] is True
        assert status["running"] is True
        assert status["service_file"] is not None

    @patch("claudegate.service._detect_platform", return_value="macos")
    @patch("claudegate.service._plist_path")
    def test_macos_not_installed(self, mock_path, _mock_plat):
        from claudegate.service import get_service_status

        mock_path.return_value.exists.return_value = False

        status = get_service_status()
        assert status["platform"] == "macos"
        assert status["installed"] is False
        assert status["running"] is False

    @patch("claudegate.service._detect_platform", return_value="linux")
    @patch("claudegate.service._systemd_unit_path")
    @patch("claudegate.service.subprocess.run")
    def test_linux_installed_active(self, mock_run, mock_path, _mock_plat):
        from claudegate.service import get_service_status

        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__str__ = lambda self: "/home/test/.config/systemd/user/claudegate.service"
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="active\n", stderr="")

        status = get_service_status()
        assert status["platform"] == "linux"
        assert status["installed"] is True
        assert status["running"] is True

    @patch("claudegate.service._detect_platform", return_value="windows")
    @patch("claudegate.service.subprocess.run")
    def test_windows_not_installed(self, mock_run, _mock_plat):
        from claudegate.service import get_service_status

        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")

        status = get_service_status()
        assert status["platform"] == "windows"
        assert status["installed"] is False
        assert status["running"] is False


class TestDashboardRoutes:
    """Tests for the dashboard HTTP endpoints."""

    @pytest.fixture
    def async_client(self):
        import httpx

        from claudegate.app import app

        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_get_dashboard_returns_html(self, async_client):
        resp = await async_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "claudegate" in resp.text
        assert "<script>" in resp.text

    async def test_api_status_returns_json(self, async_client):
        resp = await async_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()

        # Check top-level keys
        assert "health" in data
        assert "service" in data
        assert "models" in data
        assert "logs" in data

        # Check health structure
        assert "version" in data["health"]
        assert "backend" in data["health"]
        assert "status" in data["health"]

        # Check service structure
        assert "platform" in data["service"]
        assert "installed" in data["service"]
        assert "running" in data["service"]

        # Models should be a list
        assert isinstance(data["models"], list)

        # Logs should be a list
        assert isinstance(data["logs"], list)

    async def test_api_status_log_level_filter(self, async_client):
        resp = await async_client.get("/api/status?log_level=ERROR")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["logs"], list)
