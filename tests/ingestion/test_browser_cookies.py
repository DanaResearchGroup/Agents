"""Tests for browser cookie extraction and authenticated downloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.browser_cookies import (
    _domain_variants,
    _find_firefox_cookies_db,
    download_with_browser_auth,
    get_firefox_cookies,
)


# ── _domain_variants ────────────────────────────────────────────────────────


class TestDomainVariants:
    def test_standard_domain(self):
        variants = _domain_variants("pubs.acs.org")
        assert ".acs.org" in variants
        assert ".pubs.acs.org" in variants
        assert "pubs.acs.org" in variants

    def test_two_part_domain(self):
        variants = _domain_variants("example.com")
        assert ".example.com" in variants
        assert "example.com" in variants

    def test_leading_dot_stripped(self):
        variants = _domain_variants(".acs.org")
        assert ".acs.org" in variants


# ── _find_firefox_cookies_db ────────────────────────────────────────────────


class TestFindFirefoxCookiesDb:
    def test_returns_none_when_no_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert _find_firefox_cookies_db() is None

    def test_finds_cookies_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        profile = tmp_path / ".mozilla/firefox/abc123.default"
        profile.mkdir(parents=True)
        cookies_file = profile / "cookies.sqlite"
        cookies_file.write_text("")
        assert _find_firefox_cookies_db() == cookies_file


# ── get_firefox_cookies ─────────────────────────────────────────────────────


class TestGetFirefoxCookies:
    def test_no_profile_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "src.ingestion.browser_cookies._find_firefox_cookies_db",
            lambda: None,
        )
        assert get_firefox_cookies("example.com") == {}

    def test_extracts_cookies_from_db(self, tmp_path, monkeypatch):
        # Create a real SQLite cookie DB
        db_path = tmp_path / "cookies.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE moz_cookies "
            "(name TEXT, value TEXT, host TEXT)"
        )
        conn.execute(
            "INSERT INTO moz_cookies VALUES (?, ?, ?)",
            ("session_id", "abc123", ".acs.org"),
        )
        conn.execute(
            "INSERT INTO moz_cookies VALUES (?, ?, ?)",
            ("pref", "dark", ".other.com"),
        )
        conn.commit()
        conn.close()

        monkeypatch.setattr(
            "src.ingestion.browser_cookies._find_firefox_cookies_db",
            lambda: db_path,
        )
        cookies = get_firefox_cookies("pubs.acs.org")
        assert cookies == {"session_id": "abc123"}

    def test_corrupt_db_returns_empty(self, tmp_path, monkeypatch):
        db_path = tmp_path / "cookies.sqlite"
        db_path.write_text("not a database")

        monkeypatch.setattr(
            "src.ingestion.browser_cookies._find_firefox_cookies_db",
            lambda: db_path,
        )
        assert get_firefox_cookies("example.com") == {}


# ── download_with_browser_auth ──────────────────────────────────────────────


class TestDownloadWithBrowserAuth:
    @patch("src.ingestion.browser_cookies.get_firefox_cookies")
    @patch("src.ingestion.browser_cookies.requests.get")
    def test_success_pdf(self, mock_get, mock_cookies, tmp_path):
        mock_cookies.return_value = {"session": "xyz"}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF-1.4 fake content" + b"\x00" * 20_000
        mock_get.return_value = mock_response

        dest = tmp_path / "paper.pdf"
        result = download_with_browser_auth("https://pubs.acs.org/paper.pdf", dest)

        assert result == dest
        assert dest.exists()
        mock_cookies.assert_called_once_with("pubs.acs.org")

    @patch("src.ingestion.browser_cookies.get_firefox_cookies")
    @patch("src.ingestion.browser_cookies.requests.get")
    def test_403_returns_none(self, mock_get, mock_cookies, tmp_path):
        mock_cookies.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.headers = {}
        mock_get.return_value = mock_response

        dest = tmp_path / "paper.pdf"
        result = download_with_browser_auth("https://pubs.acs.org/paper.pdf", dest)
        assert result is None

    @patch("src.ingestion.browser_cookies.get_firefox_cookies")
    @patch("src.ingestion.browser_cookies.requests.get")
    def test_small_html_rejected(self, mock_get, mock_cookies, tmp_path):
        """Small HTML paywall page should not be saved as PDF."""
        mock_cookies.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}
        mock_response.content = b"<html>Please log in</html>"
        mock_get.return_value = mock_response

        dest = tmp_path / "paper.pdf"
        result = download_with_browser_auth("https://pubs.acs.org/paper.pdf", dest)
        assert result is None

    @patch("src.ingestion.browser_cookies.get_firefox_cookies")
    @patch("src.ingestion.browser_cookies.requests.get")
    def test_network_error_returns_none(self, mock_get, mock_cookies, tmp_path):
        mock_cookies.return_value = {}
        mock_get.side_effect = ConnectionError("timeout")

        dest = tmp_path / "paper.pdf"
        result = download_with_browser_auth("https://pubs.acs.org/paper.pdf", dest)
        assert result is None

    @patch("src.ingestion.browser_cookies.get_firefox_cookies")
    @patch("src.ingestion.browser_cookies.requests.get")
    def test_no_cookies_still_attempts(self, mock_get, mock_cookies, tmp_path):
        """Even with no cookies found, should still attempt download."""
        mock_cookies.return_value = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.content = b"%PDF" + b"\x00" * 20_000
        mock_get.return_value = mock_response

        dest = tmp_path / "paper.pdf"
        result = download_with_browser_auth("https://pubs.acs.org/paper.pdf", dest)
        assert result == dest
        # Verify requests.get was called with empty cookies dict
        _, kwargs = mock_get.call_args
        assert kwargs["cookies"] == {}
