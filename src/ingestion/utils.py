"""Ingestion utilities: LLM helper, file I/O, and downloads."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import requests
import yaml

if TYPE_CHECKING:
    from src.agents.llm_client import LLMClient
    from src.schemas.ingestion import RegistryResult

logger = logging.getLogger(__name__)


# ── LLM Utilities ────────────────────────────────────────────────────────────

def call_llm_sync(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    llm_client: LLMClient | None = None,
    agent_name: str | None = None,
) -> dict | None:
    """Synchronous wrapper: call LLMClient and return parsed JSON dict.

    Returns None if llm_client is None (LLM-free mode) or on failure.
    Uses asyncio.run() — acceptable for one-shot registry building calls.
    """
    if llm_client is None:
        return None

    try:
        raw = asyncio.run(llm_client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            agent_name=agent_name,
        ))
    except Exception:
        logger.exception("LLM call failed")
        return None

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON response: %s", raw[:200])
            return None
    return raw


# ── I/O Utilities ────────────────────────────────────────────────────────────

def save_registry(result: RegistryResult, outdir: str | Path) -> None:
    """Save registry results to YAML and JSON files."""
    outp = Path(outdir)
    outp.mkdir(parents=True, exist_ok=True)

    registry_data = [p.model_dump() for p in result.papers]
    with open(outp / "paper_registry.yaml", "w") as f:
        yaml.dump(registry_data, f, sort_keys=False)

    with open(outp / "paper_registry.json", "w") as f:
        json.dump(registry_data, f, indent=2)

    # Manual retrieval queue for non-OA high-priority papers.
    manual_queue = []
    for p in result.papers:
        if p.keep_for_manual_review:
            manual_queue.append({
                "id": p.id,
                "title": p.title,
                "doi": p.doi,
                "year": p.year,
                "journal": p.journal,
                "priority": p.priority,
                "reason": p.match_reason or "Direct relevance.",
                "source_url": p.source_url,
            })

    with open(outp / "manual_retrieval_queue.yaml", "w") as f:
        yaml.dump({"manual_retrieval_queue": manual_queue}, f, sort_keys=False)


# ── Download Utilities ───────────────────────────────────────────────────────

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.google.com/",
}


def download_file(
    url: str,
    outdir: Path,
    filename: str,
    use_browser_auth: bool = False,
) -> Path | None:
    """Download a file from a URL, mimicking a browser.

    If *use_browser_auth* is True and the server returns 403,
    retry using browser cookies for institutional access.
    """
    try:
        response = requests.get(
            url, headers=_DOWNLOAD_HEADERS, timeout=30,
            stream=True, allow_redirects=True,
        )

        if response.status_code == 403 and use_browser_auth:
            logger.info("Got 403, retrying with browser cookies…")
            from src.ingestion.browser_cookies import download_with_browser_auth

            dest = Path(outdir) / filename
            return download_with_browser_auth(url, dest)

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type:
            return None

        if filename.endswith(".file"):
            ext_map = {
                "application/pdf": ".pdf",
                "application/zip": ".zip",
                "text/plain": ".txt",
                "text/xml": ".xml",
            }
            for ct, ext in ext_map.items():
                if ct in content_type:
                    filename = filename.rsplit(".", 1)[0] + ext
                    break

        outdir.mkdir(parents=True, exist_ok=True)
        filepath = outdir / filename
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return filepath
    except Exception as e:
        logger.error("Failed to download %s: %s", url, e)
        return None
