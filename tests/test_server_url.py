"""Tests for server_url discovery file."""

import json
import os

from claudegate.server_url import read_server_url, remove_server_url, write_server_url


class TestWriteServerUrl:
    def test_creates_file(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        monkeypatch.setattr("claudegate.server_url.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        write_server_url("127.0.0.1", 8080)

        data = json.loads(url_file.read_text())
        assert data["url"] == "http://127.0.0.1:8080"
        assert data["pid"] == os.getpid()

    def test_creates_parent_directory(self, tmp_path, monkeypatch):
        nested = tmp_path / "sub" / "dir"
        url_file = nested / "server.json"
        monkeypatch.setattr("claudegate.server_url.CONFIG_DIR", nested)
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        write_server_url("10.0.0.1", 9090)

        assert url_file.exists()
        data = json.loads(url_file.read_text())
        assert data["url"] == "http://10.0.0.1:9090"

    def test_overwrites_existing(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        url_file.write_text('{"url": "http://old:1234", "pid": 1}')
        monkeypatch.setattr("claudegate.server_url.CONFIG_DIR", tmp_path)
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        write_server_url("127.0.0.1", 5555)

        data = json.loads(url_file.read_text())
        assert data["url"] == "http://127.0.0.1:5555"


class TestRemoveServerUrl:
    def test_removes_existing_file(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        url_file.write_text('{"url": "http://127.0.0.1:8080"}')
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        remove_server_url()

        assert not url_file.exists()

    def test_missing_file_is_fine(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        remove_server_url()  # should not raise


class TestReadServerUrl:
    def test_returns_url(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        url_file.write_text('{"url": "http://127.0.0.1:9090", "pid": 42}')
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        assert read_server_url() == "http://127.0.0.1:9090"

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        assert read_server_url() is None

    def test_malformed_json_returns_none(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        url_file.write_text("not json")
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        assert read_server_url() is None

    def test_missing_key_returns_none(self, tmp_path, monkeypatch):
        url_file = tmp_path / "server.json"
        url_file.write_text('{"pid": 42}')
        monkeypatch.setattr("claudegate.server_url.SERVER_URL_FILE", url_file)

        assert read_server_url() is None
