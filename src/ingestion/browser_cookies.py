"""Extract browser cookies for authenticated PDF downloads."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


def _find_firefox_cookies_db() -> Path | None:
    """Locate the Firefox cookies.sqlite file."""
    firefox_bases = [
        Path.home() / ".mozilla/firefox",
        Path.home() / "snap/firefox/common/.mozilla/firefox",
    ]
    for base in firefox_bases:
        if not base.exists():
            continue
        for profile_dir in base.iterdir():
            candidate = profile_dir / "cookies.sqlite"
            if candidate.exists():
                return candidate
    return None


def _domain_variants(domain: str) -> list[str]:
    """Generate domain variants for cookie matching.

    e.g. "pubs.acs.org" -> [".acs.org", ".pubs.acs.org", "pubs.acs.org"]
    """
    parts = domain.lstrip(".").split(".")
    variants = [
        f".{'.'.join(parts[i:])}"
        for i in range(len(parts) - 1)
    ]
    variants.append(domain)
    return variants


def get_firefox_cookies(domain: str) -> dict[str, str]:
    """Extract cookies for a domain from Firefox's cookie store.

    Returns dict of {name: value} for use in requests.
    """
    cookies_db = _find_firefox_cookies_db()
    if not cookies_db:
        return {}

    # Copy DB to avoid locking Firefox
    tmp_path = Path(tempfile.mktemp(suffix=".sqlite"))
    shutil.copy2(cookies_db, tmp_path)

    try:
        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()
        variants = _domain_variants(domain)
        placeholders = ",".join("?" * len(variants))
        cursor.execute(
            f"SELECT name, value FROM moz_cookies "
            f"WHERE host IN ({placeholders})",
            variants,
        )
        cookies = dict(cursor.fetchall())
        conn.close()
        return cookies
    except Exception:
        logger.debug("Failed to read Firefox cookies for %s", domain, exc_info=True)
        return {}
    finally:
        tmp_path.unlink(missing_ok=True)


def download_with_browser_auth(
    url: str,
    dest_path: Path,
    browser: str = "firefox",
) -> Path | None:
    """Download a URL using browser cookies for authentication.

    Falls back to unauthenticated download if no cookies found.
    """
    domain = urlparse(url).netloc

    cookies: dict[str, str] = {}
    if browser == "firefox":
        cookies = get_firefox_cookies(domain)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) "
            "Gecko/20100101 Firefox/120.0"
        ),
    }

    try:
        r = requests.get(
            url,
            cookies=cookies,
            headers=headers,
            allow_redirects=True,
            timeout=30,
        )
        if r.status_code == 200:
            content_type = r.headers.get("content-type", "")
            if "pdf" in content_type or len(r.content) > 10_000:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                dest_path.write_bytes(r.content)
                return dest_path
        return None
    except Exception:
        logger.debug("Browser-auth download failed for %s", url, exc_info=True)
        return None
